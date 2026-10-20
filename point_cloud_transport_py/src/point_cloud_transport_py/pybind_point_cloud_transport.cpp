// Copyright 2023 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <point_cloud_transport/subscriber.hpp>
#include <point_cloud_transport/publisher.hpp>
#include <point_cloud_transport/point_cloud_transport.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "rcl/allocator.h"
#include "rcl/arguments.h"
#include "rcl/context.h"
#include "rcl/init_options.h"

#include <rclcpp/context.hpp>
#include <rclcpp/executor.hpp>
#include <rclcpp/executor_options.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/init_options.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

#include "./pybind11.hpp"
#include "./borrowed_serialized_message.hpp"

namespace point_cloud_transport_python
{

/// The options the context a transport runs in is initialized with.
/**
 * Everything of the arguments and of the InitOptions of rclcpp that these bindings can get
 * hold of: the arguments a transport was given, which carry the remappings, the parameters
 * and the enclave of a node, the domain id to talk on, and whatever the rcl context of a
 * given rclpy context was initialized with, which is taken over as a whole.
 *
 * What is left out is left out on purpose.  The allocator of InitOptions is for rclcpp and
 * rcl to take, not the interpreter these bindings live in.  The logging of a process is set
 * up once for all by rclpy, so the context of a transport neither asks for it again - which
 * rcl would warn about - nor applies a --log-level of its own.  And no signal handler
 * belongs to these bindings either - rclpy handles Ctrl-C itself - so nothing is expected to
 * shut a context down on a signal, and nothing should do that behind the back of the
 * interpreter.
 *
 * The arguments of a context that can be read back are part of the arguments of the
 * transport, see readable_arguments_of(), but the rest of what a context was initialized with
 * is no part of this: rcl_init() keeps no copy of the arguments themselves, and everything rcl
 * made of the rest of them - the remappings, the parameter overrides, the log levels - is
 * something a new rcl_init cannot be handed.  A transport takes arguments of its own instead,
 * see the argv of PointCloudTransport.
 */
struct ContextOptions
{
  /// rcl context to take the InitOptions of rclcpp from, null for options of their own.
  const rcl_context_t * copied_from{nullptr};
  /// Domain id to talk on, none for the one the ROS environment variables point at.
  std::optional<std::size_t> domain_id;
  /// ROS arguments, the way rclcpp would have taken them from a program.
  std::vector<std::string> arguments;

  /// Order the options, so that they can be the key of a map.
  bool operator<(const ContextOptions & other) const
  {
    return std::tie(copied_from, domain_id, arguments) <
           std::tie(other.copied_from, other.domain_id, other.arguments);
  }
};

namespace
{

// Longest time the spinning thread waits for work before it checks for shutdown again.
constexpr std::chrono::milliseconds spin_timeout{10};

// The contexts the transports of this process run in, by the options they were made with.
/**
 * A context of rclcpp brings a participant of the middleware with a handful of threads of its
 * own, so transports that were made with the same options share one.  Only weak references
 * are kept, so that a context goes with the last transport that used it, and the map is
 * touched while the Python interpreter lock is held.
 */
std::map<ContextOptions, std::weak_ptr<rclcpp::Context>> contexts_by_options;

/// Return the ROS arguments a context has to be initialized with.
/**
 * \param[in] argv the arguments given to a PointCloudTransport, None for no arguments.
 * \throw std::invalid_argument if the object is not a sequence of strings.
 */
std::vector<std::string> arguments_of(const pybind11::object & argv)
{
  if (argv.is_none()) {
    return {};
  }
  // A string is a sequence of strings as well, and would come apart into single characters.
  if (pybind11::isinstance<pybind11::str>(argv) || pybind11::isinstance<pybind11::bytes>(argv)) {
    throw std::invalid_argument("argv must be a sequence of strings, not one string");
  }
  if (!pybind11::isinstance<pybind11::sequence>(argv)) {
    throw std::invalid_argument("argv must be a sequence of strings");
  }
  std::vector<std::string> arguments;
  for (const auto & argument : argv) {
    if (!pybind11::isinstance<pybind11::str>(argument)) {
      throw std::invalid_argument("every argument of argv has to be a string");
    }
    arguments.push_back(argument.cast<std::string>());
  }
  return arguments;
}

/// Check that the given object is an initialized rclpy context, and return its domain id.
/**
 * rclpy is reached through its Python interface: it says which domain id the node of a
 * transport has to talk on, and when the transport has to be released.  What it cannot give
 * is its rcl_context, because rclcpp::Context makes one itself in init() and finalizes it
 * again on its own, so a node always lives in a context of its own.  What it does hand out
 * is the address of the context it has, which is good enough to take the options it was
 * initialized with, see rcl_context_of().
 *
 * \param[in] context the object passed as the context of a PointCloudTransport.
 * \throw std::invalid_argument if the object is not an initialized rclpy context.
 */
std::size_t domain_id_of(const pybind11::object & context)
{
  const auto rclpy_context = pybind11::module_::import("rclpy.context").attr("Context");
  if (!pybind11::isinstance(context, rclpy_context)) {
    throw std::invalid_argument("context must be an rclpy.context.Context");
  }
  if (!context.attr("ok")().cast<bool>()) {
    throw std::invalid_argument("the given rclpy context is not initialized");
  }
  return context.attr("get_domain_id")().cast<std::size_t>();
}

/// Return the rcl context an rclpy context was initialized with, or null when it has none.
/**
 * Context.handle is the address of the rcl context rclpy initialized, which is as far as
 * these bindings get to the options that context was initialized with: an rclcpp::Context
 * makes its own rcl_context_t in init(), and neither rclpy nor rclcpp lets one hand the other
 * a context over.  The address is only used when it comes from the context object rclpy made
 * itself, since a context of a derived Python class can say anything here, and it is left
 * alone when an rclpy gives out something these bindings do not know.
 *
 * \param[in] context an object that is known to be an rclpy context.
 */
const rcl_context_t * rcl_context_of(const pybind11::object & context)
{
  const pybind11::object handle = pybind11::getattr(context, "handle", pybind11::none());
  const pybind11::object implementation = pybind11::getattr(
    pybind11::module_::import("rclpy.impl.implementation_singleton"), "rclpy_implementation",
    pybind11::none());
  // The class rclpy wraps its rcl context in, which is the only one to trust an address of.
  const pybind11::object handle_type = pybind11::getattr(
    implementation, "Context", pybind11::none());
  if (handle_type.is_none() || !pybind11::isinstance(handle, handle_type)) {
    return nullptr;
  }
  const pybind11::object pointer = pybind11::getattr(handle, "pointer", pybind11::none());
  if (pointer.is_none()) {
    return nullptr;
  }
  const std::uintptr_t address = pointer.cast<std::uintptr_t>();
  return 0u == address ? nullptr : reinterpret_cast<const rcl_context_t *>(address);
}

/// Return the parts of the arguments a context was initialized with that can be asked again.
/**
 * rcl_init() keeps no copy of the arguments it was given, and what it made of them is kept in
 * a form no new rcl_init() can be handed.  Two things can still be read back and asked for
 * again as arguments: the enclave, which rcl_init() takes from its arguments rather than from
 * the options it is given - left out here it would be "/" whatever the given context says -
 * and the paths of the parameter files, which are remembered as they were written.  What the
 * remappings, the parameter overrides and the log levels of a context meant stays with that
 * context, see ContextOptions for the whole of what is and is not taken over.
 *
 * \param[in] rcl_context context to ask, or null for no context.
 * \return ROS arguments, without a --ros-args marker of their own.
 */
std::vector<std::string> readable_arguments_of(const rcl_context_t * rcl_context)
{
  std::vector<std::string> arguments;
  if (nullptr == rcl_context) {
    return arguments;
  }
  const rcl_init_options_t * init_options = rcl_context_get_init_options(rcl_context);
  // rcl only hands out the options of the middleware of a context it lets you change, which
  // this one is not, so the constness has to be talked out of the way to ask for them.
  rmw_init_options_t * rmw_init_options = nullptr == init_options ? nullptr :
    rcl_init_options_get_rmw_init_options(const_cast<rcl_init_options_t *>(init_options));
  const char * enclave = nullptr == rmw_init_options ? nullptr : rmw_init_options->enclave;
  if (nullptr != enclave && std::string("/") != enclave) {
    arguments.push_back("--enclave");
    arguments.push_back(enclave);
  }

  const int count = rcl_arguments_get_param_files_count(&rcl_context->global_arguments);
  char ** parameter_files = nullptr;
  const rcl_allocator_t allocator = rcl_get_default_allocator();
  if (0 < count &&
    RCL_RET_OK == rcl_arguments_get_param_files(&rcl_context->global_arguments, allocator,
    &parameter_files))
  {
    for (int index = 0; index < count && nullptr != parameter_files; ++index) {
      arguments.push_back("--params-file");
      arguments.push_back(parameter_files[index]);
      allocator.deallocate(parameter_files[index], allocator.state);
    }
  }
  if (parameter_files) {
    allocator.deallocate(parameter_files, allocator.state);
  }
  return arguments;
}

/// Return the domain id a context has to talk on, if one was asked for.
/**
 * \param[in] domain_id the domain id given to a PointCloudTransport, None for the one of the
 *   ROS environment variables.
 * \throw std::invalid_argument if the object is not a domain id.
 */
std::optional<std::size_t> asked_domain_id(const pybind11::object & domain_id)
{
  if (domain_id.is_none()) {
    return {};
  }
  // A negative domain id is something rcl has no way to express, and one bigger than a
  // number can hold is something no domain has.
  if (!pybind11::isinstance<pybind11::int_>(domain_id) ||
    domain_id < pybind11::int_(0) ||
    domain_id > pybind11::int_(std::numeric_limits<std::int64_t>::max()))
  {
    throw std::invalid_argument("domain_id must be the number of a domain");
  }
  return static_cast<std::size_t>(domain_id.cast<std::int64_t>());
}

/// Return the options the context of a transport is made with.
/**
 * \param[in] context the rclpy context the transport follows, or None.
 * \param[in] argv the ROS arguments to initialize the context with, or None.
 * \param[in] domain_id the domain id to talk on, or None.
 * \throw std::invalid_argument for anything that is not what it has to be.
 */
ContextOptions context_options_of(
  const pybind11::object & context,
  const pybind11::object & argv,
  const pybind11::object & domain_id)
{
  ContextOptions options;
  options.arguments = arguments_of(argv);
  options.domain_id = asked_domain_id(domain_id);
  if (!context.is_none()) {
    const std::size_t domain_id_of_context = domain_id_of(context);
    if (options.domain_id && *options.domain_id != domain_id_of_context) {
      throw std::invalid_argument(
        "the given domain id " + std::to_string(*options.domain_id) + " is not the domain id " +
        std::to_string(domain_id_of_context) + " of the given context");
    }
    // Talk on the domain the given context talks on, which does not have to be the one the
    // ROS_DOMAIN_ID environment variable points at.
    options.domain_id = domain_id_of_context;
    // Initialize the context of the node with what the given one was initialized with, and
    // ask again for the parts of its arguments that can be read back.
    options.copied_from = rcl_context_of(context);
    const std::vector<std::string> inherited = readable_arguments_of(options.copied_from);
    if (!inherited.empty()) {
      // rcl always takes the first argument for the name of a program, and the arguments of
      // a transport do not have to start with one.  What the transport was given comes last,
      // so that it has the last word about anything both of them speak of, and the -- ends
      // the arguments these bindings put in, which is what rcl wants to see before anything
      // that is no argument of its own.
      if (options.arguments.empty()) {
        options.arguments.push_back("point_cloud_transport");
      }
      std::vector<std::string> arguments{"--ros-args"};
      arguments.insert(arguments.end(), inherited.begin(), inherited.end());
      arguments.push_back("--");
      options.arguments.insert(options.arguments.begin() + 1, arguments.begin(), arguments.end());
    }
  }
  return options;
}

/// Return the context of the given options, making one the first time they are needed.
/**
 * \param[in] options the context to take the options from, the domain id to talk on and the
 *   ROS arguments to initialize a fresh context with.
 * \throw anything rclcpp throws when a context cannot be initialized.
 */
std::shared_ptr<rclcpp::Context> context_for_options(const ContextOptions & options)
{
  std::weak_ptr<rclcpp::Context> & entry = contexts_by_options[options];
  if (auto context = entry.lock()) {
    return context;
  }
  std::unique_ptr<rclcpp::InitOptions> init_options;
  if (options.copied_from) {
    // Start from what the given rcl context was initialized with, so that the enclave of the
    // process and whatever else the middleware was told about it applies here as well, and
    // say something different only about the things these bindings have their own mind about.
    // rclcpp copies the given options, so the context they came from is left in one piece.
    init_options = std::make_unique<rclcpp::InitOptions>(
      *rcl_context_get_init_options(options.copied_from));
  } else {
    init_options = std::make_unique<rclcpp::InitOptions>();
  }
  // See ContextOptions for why logging is left to rclpy and signals to the interpreter.
  init_options->auto_initialize_logging(false);
  init_options->shutdown_on_signal = false;
  if (options.domain_id) {
    init_options->set_domain_id(*options.domain_id);
  }
  // rclcpp takes the arguments as a C array of strings, the way a program takes its argv.
  std::vector<const char *> arguments;
  arguments.reserve(options.arguments.size());
  for (const auto & argument : options.arguments) {
    arguments.push_back(argument.c_str());
  }
  const int argc = static_cast<int>(options.arguments.size());

  auto context = std::make_shared<rclcpp::Context>();
  context->init(argc, 0 == argc ? nullptr : arguments.data(), *init_options);
  entry = context;
  return context;
}

}  // namespace

/// The node, transport and spinning thread a PointCloudTransport is made of.
/**
 * The node runs in a context of its own rather than in the rclpy one it may have been asked
 * for: an rclpy context does not hand out its rcl_context, and rclcpp::Context makes one
 * itself and finalizes it again on its own, so there is nothing to take over.  What the node
 * does take from the rclpy context is the domain id to talk on, the moment to release
 * everything, and the options its context was initialized with, as far as rcl lets these
 * bindings read them, see ContextOptions.  The thread services the node, which is what makes
 * the callbacks of subscribed transports run, and it is the only place outside Python that
 * touches the node.
 *
 * Everything is released by shutdown(), which is also what the destructor does.  All calls
 * are made with the Python interpreter lock held, which is what keeps concurrent callers
 * from stepping on each other while shutting down.
 */
class TransportNode
{
public:
  /// Create the node, the transport and the executor servicing them, without spinning yet.
  /**
   * The thread is only started by start(), so that nothing has to be unwound when
   * constructing the transport that owns this fails afterwards.
   *
   * \param[in] node_name name of the node to create.
   * \param[in] launch_params_filepath parameters file to load, empty to load none.
   * \param[in] context initialized context to create the node in, which other transports
   *   talking on the same domain share.
   * \throw anything rclcpp throws when the node or the transport cannot be created.
   */
  TransportNode(
    const std::string & node_name,
    const std::string & launch_params_filepath,
    const std::shared_ptr<rclcpp::Context> & context)
  : context_(context)
  {
    rclcpp::NodeOptions node_options;
    node_options.context(context_);

    if (!launch_params_filepath.empty()) {
      node_options.allow_undeclared_parameters(true)
      .automatically_declare_parameters_from_overrides(true)
      .arguments({"--ros-args", "--params-file", launch_params_filepath});
    }
    node_ = rclcpp::Node::make_shared(node_name, "", node_options);

    rclcpp::ExecutorOptions executor_options;
    executor_options.context = context_;
    executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>(executor_options);
    executor_->add_node(node_);

    transport_ = std::make_unique<point_cloud_transport::PointCloudTransport>(*node_);
  }

  TransportNode(const TransportNode &) = delete;
  TransportNode & operator=(const TransportNode &) = delete;

  ~TransportNode()
  {
    shutdown();
  }

  /// Start the thread servicing the node, which runs until shutdown() is called.
  void start()
  {
    spinning_ = std::thread(
      [this]() {
        try {
          while (!stop_requested_.load()) {
            executor_->spin_all(spin_timeout);
          }
        } catch (const std::exception & e) {
          if (!stop_requested_.load()) {
            std::cerr << "Error spinning the node of a point cloud transport: " << e.what() <<
              '\n';
            // The node is not serviced any more, so say so instead of pretending it is.
            stop_requested_.store(true);
          }
        }
      });
  }

  /// End the thread, then destroy the transport, the node and the executor, at most once.
  ///
  /// The context is left to the other transports of this process that share it, and goes once
  /// nothing uses it any more; the rclpy context is left alone as well.
  ///
  /// Doing nothing when this was already done makes it safe to call from Python, from the
  /// destructor, and from the shutdown callback of an rclpy context, in any order.
  void shutdown()
  {
    if (is_shutdown_) {
      return;
    }
    is_shutdown_ = true;

    // The thread notices within one spin timeout, and never touches Python itself.
    stop_requested_.store(true);
    if (spinning_.joinable()) {
      // Let the interpreter go while waiting for the thread to end.
      pybind11::gil_scoped_release release;
      spinning_.join();
    }
    transport_.reset();
    node_.reset();
    executor_.reset();
    context_.reset();
  }

  /// Advertise a point cloud topic on the node of this transport.
  /**
   * \throw std::runtime_error if this transport was shut down already.
   */
  point_cloud_transport::Publisher advertise(const std::string & base_topic, uint32_t queue_size)
  {
    if (!transport_) {
      throw std::runtime_error("this PointCloudTransport has been shut down");
    }
    return transport_->advertise(base_topic, queue_size);
  }

  /// Return whether the node is still there and being serviced.
  bool ok() const
  {
    return transport_ && !stop_requested_.load() && context_ && context_->is_valid();
  }

  /// Return the name of the node this transport runs.
  std::string name() const
  {
    return node_ ? node_->get_name() : std::string();
  }

private:
  std::atomic_bool stop_requested_{false};
  std::shared_ptr<rclcpp::Context> context_;
  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  std::unique_ptr<point_cloud_transport::PointCloudTransport> transport_;
  std::thread spinning_;
  bool is_shutdown_ = false;
};

/// The bindings of the point_cloud_transport::PointCloudTransport class.
/**
 * Owns the node, transport and thread of one instance, and follows the life cycle of the
 * rclpy context it was created with, if any.
 */
class PointCloudTransport
{
public:
  /// Create the node, the transport and the thread servicing them.
  /**
   * \param[in] node_name name of the node to create.
   * \param[in] launch_params_filepath parameters file to load, empty to load none.
   * \param[in] context optional initialized rclpy context to follow.
   * \param[in] argv optional ROS arguments to initialize the context of the node with.
   * \param[in] domain_id optional domain id for the node to talk on.
   * \throw std::invalid_argument if the context, the arguments or the domain id are not what
   *   they have to be.
   * \throw anything rclcpp throws when the node cannot be created.
   */
  PointCloudTransport(
    const std::string & node_name,
    const std::string & launch_params_filepath,
    pybind11::object context,
    const pybind11::object & argv,
    const pybind11::object & domain_id)
  : context_(std::move(context))
  {
    const ContextOptions options = context_options_of(context_, argv, domain_id);
    transport_ = std::make_shared<TransportNode>(
      node_name, launch_params_filepath, context_for_options(options));

    if (!context_.is_none()) {
      // Release the transport as soon as the given context is shut down, without having to
      // ask for it.  The callback only holds a weak reference, so it stays harmless once
      // this transport is gone, and the context is kept alive while it is registered here.
      std::weak_ptr<TransportNode> weak_transport = transport_;
      context_.attr("on_shutdown")(pybind11::cpp_function(
          [weak_transport]() {
            auto transport = weak_transport.lock();
            if (transport) {
              transport->shutdown();
            }
          }));
    }
    transport_->start();
  }

  PointCloudTransport(const PointCloudTransport &) = delete;
  PointCloudTransport & operator=(const PointCloudTransport &) = delete;

  ~PointCloudTransport()
  {
    try {
      shutdown();
    } catch (const std::exception & e) {
      std::cerr << "Error cleaning up a point cloud transport: " << e.what() << '\n';
    }
  }

  /// Advertise a point cloud topic, and return the publisher to publish on it.
  point_cloud_transport::Publisher advertise(const std::string & base_topic, uint32_t queue_size)
  {
    return running().advertise(base_topic, queue_size);
  }

  /// End the thread servicing the node, and destroy the node.
  ///
  /// The transport is not usable afterwards, and calling this again does nothing.
  void shutdown()
  {
    // Hand the transport over to this scope, so that it stays alive while its thread is
    // waited for with the interpreter lock released, whatever another thread does while
    // waiting.
    auto transport = std::move(transport_);
    context_ = pybind11::none();
    if (transport) {
      transport->shutdown();
    }
  }

  /// Return whether the transport can still be used.
  bool ok() const
  {
    return transport_ && transport_->ok();
  }

  /// Return the name of the node the transport runs.
  std::string name() const
  {
    return transport_ ? transport_->name() : std::string();
  }

private:
  /// Return the transport, which has to be alive to be used.
  /**
   * \throw std::runtime_error if this transport was shut down already.
   */
  TransportNode & running()
  {
    if (!transport_) {
      throw std::runtime_error("this PointCloudTransport has been shut down");
    }
    return *transport_;
  }

  std::shared_ptr<TransportNode> transport_;
  pybind11::object context_ = pybind11::none();
};

// Bindings for the PointCloudCodec class
PYBIND11_MODULE(_point_cloud_transport, m)
{
  m.doc() = "Python wrapper of the point_cloud_transport API";

  pybind11::class_<point_cloud_transport::Publisher>(m, "Publisher")
  .def(pybind11::init())
  .def("get_topic",
      &point_cloud_transport::Publisher::getTopic,
      "Returns the base point cloud topic.")
  .def("get_num_subscribers",
      &point_cloud_transport::Publisher::getNumSubscribers,
      "Returns the number of subscribers this publisher is connected to.")
  .def("shutdown", &point_cloud_transport::Publisher::shutdown,
      "Unsubscribe the callback associated with this Publisher.")
  .def("publish", [](point_cloud_transport::Publisher & publisher,
    const pybind11::buffer & buffer) {
      // Deserialize straight from the Python memory, without intermediate copies.
      sensor_msgs::msg::PointCloud2 pc2;
      {
        point_cloud_transport_python::BorrowedSerializedMessage borrowed(buffer);
        rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serialization;
        serialization.deserialize_message(&borrowed.get(), &pc2);
      }
      publisher.publish(pc2);
       },
       "Publish a serialized sensor_msgs/PointCloud2 (bytes) on the topics associated with this "
       "Publisher.");

  pybind11::class_<point_cloud_transport_python::PointCloudTransport>(
    m, "PointCloudTransport",
    "Advertise and subscribe to point clouds on a node of its own.\n\n"
    "The node runs in a context of its own, which the transports created with the same options\n"
    "share, and is serviced by a thread that this instance owns.  Call shutdown(), or use the\n"
    "transport as a context manager, to end that thread and destroy the node.  A transport\n"
    "created with an rclpy context is initialized with the options of that context, and does\n"
    "that cleanup by itself once the context is shut down.")
  .def(
    pybind11::init(
      [](const std::string & node_name,
      const std::string & launch_params_filepath,
      pybind11::object context,
      pybind11::object argv,
      pybind11::object domain_id) {
        return std::make_unique<point_cloud_transport_python::PointCloudTransport>(
          node_name, launch_params_filepath, std::move(context), argv, domain_id);
      }),
    pybind11::arg("node_name"),
    pybind11::arg("launch_params_filepath") = std::string(),
    pybind11::arg("context") = pybind11::none(),
    pybind11::arg("argv") = pybind11::none(),
    pybind11::arg("domain_id") = pybind11::none(),
    "Create the node and the transport running on it.\n\n"
    "\\:param node_name: name of the node to create.\n"
    "\\:param launch_params_filepath: parameters file to configure the node with, empty for\n"
    "  none.\n"
    "\\:param context: optional rclpy.context.Context, which has to be initialized.  The\n"
    "  context of the node is initialized with the options of the given context as far as\n"
    "  rclcpp lets them be taken, its node gets the parameter files that context was told\n"
    "  about, the node talks on the domain id of that context, and this transport is cleaned\n"
    "  up as soon as the context is shut down.  Defaults to None, which makes the transport\n"
    "  live until it is shut down by hand.\n"
    "\\:param argv: optional sequence of command line arguments to initialize the context of\n"
    "  the node with, the way rclcpp takes them from a program, e.g. sys.argv.  They carry\n"
    "  the remappings, the parameters and the enclave of the node, and go behind what could\n"
    "  be read back from the given context, which keeps no copy of its own arguments; a log\n"
    "  level among them reaches no node, as rclpy sets the logging of the process up for\n"
    "  good.  Defaults to None, which gives the node no ROS arguments of its own.\n"
    "\\:param domain_id: optional domain id for the node to talk on, which has to be the one\n"
    "  of the given context.  Defaults to None, which takes the domain id of the given\n"
    "  context or the one of the ROS environment variables.\n")
  .def(
    "advertise",
    &point_cloud_transport_python::PointCloudTransport::advertise,
    pybind11::arg("base_topic"),
    pybind11::arg("queue_size"),
    "Advertise a sensor_msgs/PointCloud2 topic, and return the Publisher to publish on it.")
  .def(
    "shutdown",
    &point_cloud_transport_python::PointCloudTransport::shutdown,
    "End the thread servicing the node, and destroy the node.")
  .def(
    "ok",
    &point_cloud_transport_python::PointCloudTransport::ok,
    "Return whether this transport can still be used.")
  .def(
    "name",
    &point_cloud_transport_python::PointCloudTransport::name,
    "Return the name of the node this transport runs, or an empty string once shut down.")
  .def(
    "__enter__",
    [](pybind11::object transport) {
      return transport;
    },
    "Return the transport itself, so it can be used with a with statement.")
  .def(
    "__exit__",
    [](point_cloud_transport_python::PointCloudTransport & transport,
    pybind11::args) {
      transport.shutdown();
    },
    "Shut the transport down when the with statement ends.");
  // TODO(ahcorde): Revisit this method
  // .def("subscribe",
  //      [](point_cloud_transport::PointCloudTransport & pct,
  //         const std::string & base_topic,
  //         uint32_t queue_size,
  //         const std::function<void(const char *)> & callback)
  //    {
  //      return pct.subscribe_lol(
  //        base_topic, queue_size,
  //        [callback](const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg)
  //        {
  //          std::cerr << "1" << '\n';
  //          rclcpp::Serialization<sensor_msgs::msg::PointCloud2> serialization;
  //          rclcpp::SerializedMessage serialized_msg;
  //
  //          serialization.serialize_message(msg.get(), &serialized_msg);
  //
  //          std::string buffer;
  //          buffer.resize(serialized_msg.get_rcl_serialized_message().buffer_capacity);
  //          std::cerr << "buffer_capacity: "
  //                    << serialized_msg.get_rcl_serialized_message().buffer_capacity
  //                    << '\n';
  //          memcpy(
  //            buffer.data(),
  //            serialized_msg.get_rcl_serialized_message().buffer,
  //            serialized_msg.get_rcl_serialized_message().buffer_capacity);
  //
  //          std::cerr << "2" << '\n';
  //          callback(buffer.c_str());
  //          std::cerr << "3" << '\n';
  //        });
  //    });
  // ;

  pybind11::class_<point_cloud_transport::Subscriber>(m, "Subscriber")
  .def(pybind11::init())
  .def("get_topic",
       &point_cloud_transport::Subscriber::getTopic,
       "Returns the base point cloud topic.")
  .def("get_num_publishers",
       &point_cloud_transport::Subscriber::getNumPublishers,
       "Returns the number of publishers this subscriber is connected to.")
  .def("get_transport",
       &point_cloud_transport::Subscriber::getTransport,
       "Returns the name of the transport being used.")
  .def("shutdown", &point_cloud_transport::Subscriber::shutdown,
       "Unsubscribe the callback associated with this Subscriber.");
}
}  // namespace point_cloud_transport_python
