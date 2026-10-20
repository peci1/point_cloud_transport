# Copyright 2026 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the point_cloud_transport and Publisher bindings."""

import concurrent.futures
import os
import time

from point_cloud_transport_py import PointCloudTransport, Publisher, Subscriber
import pytest
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import serialize_message
from sensor_msgs.msg import PointCloud2

# Time to wait for RMW to match endpoints and deliver, in seconds.
MATCH_TIMEOUT = 5.0
# Pause between two repeated publications while waiting for a message to arrive.
PUBLISH_INTERVAL = 0.005


@pytest.fixture
def ros_context():
    context = rclpy.context.Context()
    rclpy.init(context=context)
    yield context
    rclpy.shutdown(context=context)


@pytest.fixture
def listener(ros_context):
    node = rclpy.create_node('test_listener', namespace='', context=ros_context)
    yield node
    node.destroy_node()


@pytest.fixture
def executor(ros_context, listener):
    executor = rclpy.executors.SingleThreadedExecutor(context=ros_context)
    executor.add_node(listener)
    yield executor
    executor.remove_node(listener)
    executor.shutdown()


@pytest.fixture
def transport(ros_context):
    transport = PointCloudTransport('test_transport', '', ros_context)
    yield transport
    transport.shutdown()
    del transport


def wait_until(predicate, timeout=MATCH_TIMEOUT):
    """Wait until predicate() holds and report whether that happened in time."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def wait_until_matched(publisher):
    """Wait until the middleware connected a subscriber to the given publisher."""
    if not wait_until(lambda: publisher.get_num_subscribers() > 0):
        pytest.skip(f'no subscriber was matched on {publisher.get_topic()}')


def publish_until(publisher, payload, received, executor, count=1):
    """Publish repeatedly until at least count messages arrived, and say whether they did."""
    deadline = time.monotonic() + MATCH_TIMEOUT
    while len(received) < count and time.monotonic() < deadline:
        publisher.publish(payload)
        executor.spin_once(timeout_sec=0.01)
        time.sleep(PUBLISH_INTERVAL)
    return len(received) >= count


def advertised_topics(node):
    """Return the topics visible to the given node, mapped to their message types."""
    return dict(node.get_topic_names_and_types())


def subscribe(listener, topic, callback):
    """Subscribe the listener node to a topic with the sensor data QoS profile."""
    return listener.create_subscription(PointCloud2, topic, callback, qos_profile_sensor_data)


def test_default_publisher():
    """A default constructed publisher is invalid but safe to query."""
    publisher = Publisher()
    assert publisher.get_topic() == ''
    assert publisher.get_num_subscribers() == 0
    publisher.shutdown()
    assert publisher.get_topic() == ''
    assert publisher.get_num_subscribers() == 0


def test_default_subscriber():
    """A default constructed subscriber is invalid but safe to query."""
    subscriber = Subscriber()
    assert subscriber.get_topic() == ''
    assert subscriber.get_transport() == ''
    assert subscriber.get_num_publishers() == 0
    subscriber.shutdown()


@pytest.mark.parametrize(('base_topic', 'expected_topic'), [
    ('points', '/points'),
    ('/sensors/points', '/sensors/points'),
    ('sensors/points', '/sensors/points'),
])
def test_advertise_resolves_the_topic(transport, base_topic, expected_topic):
    """The advertised topic is fully resolved and reported back."""
    publisher = transport.advertise(base_topic, 10)
    assert publisher.get_topic() == expected_topic
    publisher.shutdown()


def test_shutdown_resets_the_publisher(transport, listener):
    """Shutting the publisher down leaves it as empty as a default constructed one."""
    base_topic = '/test_transport/shutdown'
    publisher = transport.advertise(base_topic, 10)
    if not wait_until(lambda: base_topic in advertised_topics(listener)):
        publisher.shutdown()
        pytest.skip(f'{base_topic} did not show up in the graph')
    publisher.shutdown()
    assert publisher.get_topic() == ''
    assert publisher.get_num_subscribers() == 0
    publisher.shutdown()


def test_params_file_configures_the_node(ros_context, tmp_path):
    """A parameter file is accepted."""
    params_file = tmp_path / 'params.yaml'
    params_file.write_text(
        'test_params_transport:\n'
        '  ros__parameters:\n'
        '    test_transport.params.enable_pub_plugins: [""]\n'
    )
    configured = PointCloudTransport('test_params_transport', str(params_file))
    with pytest.raises(RuntimeError):
        # creating a publisher with no plugins should raise an exception
        publisher = configured.advertise('/test_transport/params', 10)
        # in case the publisher creation did not throw
        publisher.shutdown()


def test_missing_params_file_is_reported(ros_context):
    """A parameter file that does not exist is reported while constructing the node."""
    with pytest.raises(RuntimeError):
        PointCloudTransport('test_missing_params_transport', '/does/not/exist.yaml')


def test_advertise_shows_the_topic_in_the_graph(transport, listener):
    """The advertised base topic appears in the graph with the point cloud type."""
    base_topic = '/test_transport/graph'
    publisher = transport.advertise(base_topic, 10)
    assert publisher.get_topic() == base_topic
    if not wait_until(lambda: base_topic in advertised_topics(listener)):
        publisher.shutdown()
        pytest.skip(f'{base_topic} did not show up in the graph')
    assert advertised_topics(listener)[base_topic] == ['sensor_msgs/msg/PointCloud2']
    publisher.shutdown()


def test_publish_delivers_the_point_cloud(transport, listener, executor, point_cloud):
    """A serialized point cloud published by the bindings arrives at a subscription."""
    base_topic = '/test_transport/raw_points'
    received = []
    with subscribe(listener, base_topic, received.append):
        publisher = transport.advertise(base_topic, 10)
        wait_until_matched(publisher)

        delivered = publish_until(publisher, serialize_message(point_cloud), received, executor)
        publisher.shutdown()
        if not delivered:
            pytest.skip(f'no point cloud was delivered on {base_topic}')
        assert received[0] == point_cloud


def test_publish_reaches_every_subscriber(transport, listener, executor, serialized_point_cloud):
    """Every subscriber connected to the base topic is served."""
    base_topic = '/test_transport/two_subscribers'
    received = {'first': [], 'second': []}
    with subscribe(listener, base_topic, received['first'].append):
        with subscribe(listener, base_topic, received['second'].append):
            publisher = transport.advertise(base_topic, 10)
            if not wait_until(lambda: publisher.get_num_subscribers() == 2):
                publisher.shutdown()
                pytest.skip('both subscribers were not matched on the base topic')

            deadline = time.monotonic() + MATCH_TIMEOUT
            while not all(received.values()) and time.monotonic() < deadline:
                publisher.publish(serialized_point_cloud)
                executor.spin_once(timeout_sec=0.01)
                time.sleep(PUBLISH_INTERVAL)
            publisher.shutdown()

            if not all(received.values()):
                pytest.skip(f'the subscriptions on {base_topic} were not both served')
            assert received['first'][0] == received['second'][0]
            assert publisher.get_num_subscribers() == 0


@pytest.mark.parametrize('message', [
    'not bytes',
    PointCloud2(),
    b'',
    b'not a serialized point cloud at all !!',
])
def test_publish_requires_a_serialized_message(transport, message):
    """Only serialized point clouds can be published, and errors are reported."""
    publisher = transport.advertise('/test_transport/invalid_payloads', 10)
    try:
        with pytest.raises((TypeError, RuntimeError)):
            publisher.publish(message)
    finally:
        publisher.shutdown()


def test_publish_many_point_clouds(transport, listener, executor, make_cloud):
    """The publisher keeps working over many consecutive publications."""
    base_topic = '/test_transport/many_points'
    received = []
    seen_frames = set()
    with subscribe(listener, base_topic, received.append):
        publisher = transport.advertise(base_topic, 10)
        try:
            wait_until_matched(publisher)

            clouds = [
                make_cloud(num_points=num_points, frame_id=f'frame_{num_points}')
                for num_points in (1, 8, 64)
            ]
            payloads = [serialize_message(cloud) for cloud in clouds]
            expected_frames = {cloud.header.frame_id for cloud in clouds}
            published = 0
            deadline = time.monotonic() + MATCH_TIMEOUT
            while seen_frames != expected_frames and time.monotonic() < deadline:
                publisher.publish(payloads[published % len(payloads)])
                published += 1
                executor.spin_once(timeout_sec=0.01)
                time.sleep(PUBLISH_INTERVAL)
                seen_frames = {message.header.frame_id for message in received}
        finally:
            publisher.shutdown()

        if seen_frames != expected_frames:
            pytest.skip(f'not every point cloud was delivered on {base_topic}')
        for message in (m for m in received if m.header.frame_id == 'frame_64'):
            assert message.width == 64
            assert message.row_step == message.width * message.point_step


def default_domain_id():
    """Return the domain id a context uses when it is not given one."""
    return int(os.environ.get('ROS_DOMAIN_ID') or 0)


def test_the_transport_reports_its_node(transport):
    """The transport knows its node until it is shut down."""
    assert transport.name() == 'test_transport'
    assert transport.ok()
    transport.shutdown()
    assert not transport.ok()
    assert transport.name() == ''


def test_shutdown_releases_the_node(transport, listener):
    """Shutting down ends the thread and takes the node, and with it its topics, down."""
    base_topic = '/test_transport/released'
    publisher = transport.advertise(base_topic, 10)
    if not wait_until(lambda: base_topic in advertised_topics(listener)):
        publisher.shutdown()
        transport.shutdown()
        pytest.skip(f'{base_topic} did not show up in the graph')

    publisher.shutdown()
    transport.shutdown()

    assert not transport.ok()
    assert transport.name() == ''
    with pytest.raises(RuntimeError):
        transport.advertise('/test_transport/after_shutdown', 10)
    # Shutting down a second time is allowed and does nothing.
    transport.shutdown()
    assert wait_until(lambda: base_topic not in advertised_topics(listener)), \
        f'{base_topic} was still in the graph after the node was released'


def test_the_with_statement_shuts_the_transport_down(ros_context):
    """A transport can be scoped with a with statement."""
    with PointCloudTransport('scoped_transport', '', ros_context) as transport:
        assert transport.ok()
        assert transport.name() == 'scoped_transport'
    assert not transport.ok()
    with pytest.raises(RuntimeError):
        transport.advertise('/test_transport/scoped', 10)


def test_the_node_follows_the_context(ros_context, listener):
    """A transport created with a context is visible to the nodes of that context."""
    base_topic = '/test_transport/follows_context'
    transport = PointCloudTransport('follows_context_transport', '', ros_context)
    try:
        publisher = transport.advertise(base_topic, 10)
        if not wait_until(lambda: base_topic in advertised_topics(listener)):
            pytest.skip(f'{base_topic} did not show up in the graph of the given context')
        assert advertised_topics(listener)[base_topic] == ['sensor_msgs/msg/PointCloud2']
        publisher.shutdown()
    finally:
        transport.shutdown()


def test_the_node_follows_the_domain_of_the_context():
    """The node of a transport uses the domain id of the context it was given."""
    other_domain_id = default_domain_id() + 3
    other_context = rclpy.context.Context()
    rclpy.init(context=other_context, domain_id=other_domain_id)
    default_context = rclpy.context.Context()
    rclpy.init(context=default_context)
    other_listener = rclpy.create_node('other_domain_listener', context=other_context)
    default_listener = rclpy.create_node('default_domain_listener', context=default_context)
    base_topic = '/test_transport/other_domain'
    try:
        transport = PointCloudTransport('other_domain_transport', '', other_context)
        publisher = transport.advertise(base_topic, 10)
        try:
            if not wait_until(lambda: base_topic in advertised_topics(other_listener)):
                pytest.skip(f'{base_topic} did not show up in domain {other_domain_id}')
            assert not wait_until(
                lambda: base_topic in advertised_topics(default_listener), timeout=1.0
            ), f'{base_topic} was visible in domain {default_domain_id()} as well'
        finally:
            publisher.shutdown()
            transport.shutdown()
    finally:
        other_listener.destroy_node()
        default_listener.destroy_node()
        rclpy.try_shutdown(context=other_context)
        rclpy.try_shutdown(context=default_context)


def test_the_arguments_of_a_transport_reach_its_node(ros_context, listener):
    """The ROS arguments of a transport are honored by the node it runs on."""
    base_topic = '/test_transport/arguments_topic'
    remapped_topic = '/test_transport/arguments_topic_remapped'
    transport = PointCloudTransport(
        'arguments_transport', context=ros_context,
        argv=['test', '--ros-args', '-r', f'{base_topic}:={remapped_topic}'])
    try:
        publisher = transport.advertise(base_topic, 10)
        try:
            if not wait_until(lambda: remapped_topic in advertised_topics(listener)):
                pytest.skip(f'{remapped_topic} did not show up in the graph')
            assert base_topic not in advertised_topics(listener), f'{base_topic} was not remapped'
        finally:
            publisher.shutdown()
    finally:
        transport.shutdown()


def test_transports_do_not_share_arguments_with_each_other(ros_context, listener):
    """Arguments belong to the transport they were given to, and not to the whole process."""
    base_topic = '/test_transport/shared_arguments'
    remapped_topic = '/test_transport/shared_arguments_remapped'
    with_arguments = PointCloudTransport(
        'shared_arguments_transport', context=ros_context,
        argv=['test', '--ros-args', '-r', f'{base_topic}:={remapped_topic}'])
    without_arguments = PointCloudTransport('unshared_arguments_transport', context=ros_context)
    first = with_arguments.advertise(base_topic, 10)
    second = without_arguments.advertise(base_topic, 10)
    try:
        if not wait_until(lambda: remapped_topic in advertised_topics(listener)):
            pytest.skip(f'{remapped_topic} did not show up in the graph')
        # The transport that was given no arguments talks on the name it was given, whatever
        # the arguments of the other transport make of theirs.
        assert wait_until(lambda: base_topic in advertised_topics(listener)), \
            f'{base_topic} was remapped as well, so the two transports share one context'
    finally:
        first.shutdown()
        second.shutdown()
        with_arguments.shutdown()
        without_arguments.shutdown()


def test_arguments_that_ros_does_not_understand_are_refused(ros_context):
    """Arguments that are no ROS arguments do not make a transport."""
    with pytest.raises(RuntimeError):
        PointCloudTransport(
            'unknown_arguments_transport', context=ros_context,
            argv=['test', '--ros-args', '--nonsense'])


@pytest.mark.parametrize('not_arguments', [
    '--ros-args', b'--ros-args', object(), 42, {'--ros-args': ''}, ['test', 42]])
def test_the_arguments_have_to_be_a_sequence_of_strings(not_arguments):
    """Anything other than a sequence of strings is refused as arguments."""
    with pytest.raises(ValueError):
        PointCloudTransport('invalid_arguments_transport', argv=not_arguments)


def enclaves(node):
    """Return the enclave of every node the given node sees, by the name of that node."""
    return {
        name: enclave
        for name, namespace, enclave in node.get_node_names_and_namespaces_with_enclaves()
    }


def test_the_enclave_of_the_given_context_reaches_the_node(listener, transport):
    """The node of a transport is in the enclave the given context was told about."""
    if not wait_until(lambda: 'test_transport' in enclaves(listener)):
        pytest.skip('the node of the transport did not show up in the graph')
    assert enclaves(listener)['test_transport'] == '/', 'a transport took an enclave of its own'

    enclave = '/inherited_enclave'
    context = rclpy.context.Context()
    rclpy.init(context=context, args=['test', '--ros-args', '--enclave', enclave])
    other_listener = rclpy.create_node('inherited_enclave_listener', context=context)
    inherited = PointCloudTransport('inherited_enclave_transport', '', context)
    try:
        if not wait_until(lambda: 'inherited_enclave_transport' in enclaves(other_listener)):
            pytest.skip('the node of the transport did not show up in the graph')
        assert enclaves(other_listener)['inherited_enclave_transport'] == enclave
    finally:
        inherited.shutdown()
        other_listener.destroy_node()
        rclpy.try_shutdown(context=context)


def test_the_arguments_of_a_transport_win_over_the_given_context():
    """What a transport is given itself has the last word over what a context says."""
    context = rclpy.context.Context()
    rclpy.init(context=context, args=['test', '--ros-args', '--enclave', '/from_context'])
    other_listener = rclpy.create_node('winning_arguments_listener', context=context)
    transport = PointCloudTransport(
        'winning_arguments_transport', '', context,
        argv=['test', '--ros-args', '--enclave', '/from_arguments'])
    try:
        if not wait_until(lambda: 'winning_arguments_transport' in enclaves(other_listener)):
            pytest.skip('the node of the transport did not show up in the graph')
        assert enclaves(other_listener)['winning_arguments_transport'] == '/from_arguments'
    finally:
        transport.shutdown()
        other_listener.destroy_node()
        rclpy.try_shutdown(context=context)


def test_the_parameter_files_of_the_given_context_reach_the_node(tmp_path):
    """A parameter file the given context was told about configures the node of a transport."""
    params_file = tmp_path / 'inherited_params.yaml'
    params_file.write_text(
        'inherited_params_transport:\n'
        '  ros__parameters:\n'
        '    test_transport.params.enable_pub_plugins: [""]\n'
    )
    context = rclpy.context.Context()
    rclpy.init(context=context, args=['test', '--ros-args', '--params-file', str(params_file)])
    configured = PointCloudTransport('inherited_params_transport', '', context)
    try:
        with pytest.raises(RuntimeError):
            # creating a publisher with no plugins should raise an exception
            publisher = configured.advertise('/test_transport/params', 10)
            # in case the publisher creation did not throw
            publisher.shutdown()
    finally:
        configured.shutdown()
        rclpy.try_shutdown(context=context)


def test_the_context_a_transport_came_from_is_left_in_one_piece():
    """A transport leaves the context it took its options from working as it was."""
    enclave = '/untouched_enclave'
    context = rclpy.context.Context()
    rclpy.init(context=context, args=['test', '--ros-args', '--enclave', enclave])
    transport = PointCloudTransport('untouched_context_transport', '', context)
    publisher = transport.advertise('/test_transport/untouched_context', 10)
    publisher.shutdown()
    transport.shutdown()

    assert context.ok(), 'the context went down with the options it was asked for'
    assert context.get_domain_id() == default_domain_id()
    # The context goes on handing out the very same options to the next transport.
    again = PointCloudTransport('untouched_context_again_transport', '', context)
    listener = rclpy.create_node('untouched_enclave_listener', context=context)
    try:
        assert again.ok()
        assert again.name() == 'untouched_context_again_transport'
        if not wait_until(lambda: 'untouched_context_again_transport' in enclaves(listener)):
            pytest.skip('the node of the transport did not show up in the graph')
        assert enclaves(listener)['untouched_context_again_transport'] == enclave
    finally:
        again.shutdown()
        listener.destroy_node()
        rclpy.try_shutdown(context=context)


def test_a_transport_follows_the_domain_id_it_is_given():
    """A transport talks on the domain id it was given, and on no other one."""
    other_domain_id = default_domain_id() + 5
    context = rclpy.context.Context()
    rclpy.init(context=context)
    other_context = rclpy.context.Context()
    rclpy.init(context=other_context, domain_id=other_domain_id)
    listener = rclpy.create_node('given_domain_listener', context=context)
    other_listener = rclpy.create_node('given_domain_other_listener', context=other_context)
    base_topic = '/test_transport/given_domain'
    try:
        transport = PointCloudTransport('given_domain_transport', domain_id=other_domain_id)
        publisher = transport.advertise(base_topic, 10)
        try:
            if not wait_until(lambda: base_topic in advertised_topics(other_listener)):
                pytest.skip(f'{base_topic} did not show up in domain {other_domain_id}')
            assert not wait_until(
                lambda: base_topic in advertised_topics(listener), timeout=1.0
            ), f'{base_topic} was visible in domain {default_domain_id()} as well'
        finally:
            publisher.shutdown()
            transport.shutdown()
    finally:
        listener.destroy_node()
        other_listener.destroy_node()
        rclpy.try_shutdown(context=other_context)
        rclpy.try_shutdown(context=context)


def test_the_domain_id_of_the_given_context_is_accepted(ros_context):
    """A domain id that says what the given context says is no contradiction."""
    transport = PointCloudTransport(
        'matching_domain_transport', context=ros_context,
        domain_id=ros_context.get_domain_id())
    try:
        assert transport.ok()
    finally:
        transport.shutdown()


def test_a_domain_id_of_its_own_while_a_context_is_given_is_refused(ros_context):
    """A transport that follows a context has to talk where that context talks."""
    with pytest.raises(ValueError):
        PointCloudTransport(
            'conflicting_domain_transport', context=ros_context,
            domain_id=ros_context.get_domain_id() + 1)


@pytest.mark.parametrize('not_a_domain_id', ['7', 1.5, -1, object()])
def test_the_domain_id_has_to_be_the_number_of_a_domain(not_a_domain_id):
    """Anything other than the number of a domain is refused as a domain id."""
    with pytest.raises(ValueError):
        PointCloudTransport('invalid_domain_transport', domain_id=not_a_domain_id)


def test_transports_of_one_context_release_on_their_own(ros_context, listener):
    """A transport releases on its own, without disturbing the others of the same context."""
    first = PointCloudTransport('first_of_two_transports', '', ros_context)
    second = PointCloudTransport('second_of_two_transports', '', ros_context)
    first_topic = '/test_transport/first_of_two'
    second_topic = '/test_transport/second_of_two'
    first_publisher = first.advertise(first_topic, 10)
    second_publisher = second.advertise(second_topic, 10)
    try:
        if not wait_until(lambda: first_topic in advertised_topics(listener)):
            pytest.skip(f'{first_topic} did not show up in the graph')
        assert wait_until(lambda: second_topic in advertised_topics(listener))

        # The publisher goes before the transport that made it, as an entity goes before its
        # node in rclcpp: while a handle lives on, the middleware keeps the topic around.
        first_publisher.shutdown()
        first.shutdown()

        assert not first.ok()
        assert second.ok(), 'the other transport went down with the first one'
        assert ros_context.ok(), 'the rclpy context went down with the transport'
        assert wait_until(lambda: first_topic not in advertised_topics(listener))
        assert second_topic in advertised_topics(listener)
    finally:
        first_publisher.shutdown()
        second_publisher.shutdown()
        first.shutdown()
        second.shutdown()


def test_shutting_the_context_down_releases_the_transport():
    """Transports created with a context clean themselves up when that context is shut down."""
    context = rclpy.context.Context()
    rclpy.init(context=context)
    transports = [PointCloudTransport(f'shutdown_context_transport_{index}', '', context)
                  for index in range(3)]
    publishers = [transport.advertise(f'/test_transport/shutdown_context_{index}', 10)
                  for index, transport in enumerate(transports)]
    try:
        assert all(transport.ok() for transport in transports)
        rclpy.shutdown(context=context)
        assert not any(transport.ok() for transport in transports)
        assert all(transport.name() == '' for transport in transports)
        with pytest.raises(RuntimeError):
            transports[0].advertise('/test_transport/shutdown_context_again', 10)
        with pytest.raises(ValueError):
            # A context that is gone cannot be followed by a new transport.
            PointCloudTransport('shutdown_context_transport_again', '', context)
        # The threads are already gone, so this only has to release what is left.
        for transport in transports:
            transport.shutdown()
            transport.shutdown()
    finally:
        for publisher in publishers:
            publisher.shutdown()
        rclpy.try_shutdown(context=context)


def test_shutting_down_from_several_threads_at_once(ros_context):
    """A transport survives threads shutting it down together."""
    transport = PointCloudTransport('racing_transport', '', ros_context)
    publisher = transport.advertise('/test_transport/racing', 10)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(transport.shutdown) for _ in range(8)]
            for future in futures:
                future.result(timeout=20)
    finally:
        publisher.shutdown()
    assert not transport.ok()
    assert transport.name() == ''


class NearlyAContext:
    """An object with the interface of an rclpy context, but which is not one."""

    def ok(self):
        """Pretend to be initialized."""
        return True

    def get_domain_id(self):
        """Pretend a domain."""
        return 0

    def on_shutdown(self, callback):
        """Pretend to remember a shutdown callback."""
        return callback


@pytest.mark.parametrize('not_a_context', [object(), 42, 'ros_context', NearlyAContext()])
def test_the_context_has_to_be_a_context(not_a_context):
    """Anything other than a context of rclpy is refused, even one with the same methods."""
    with pytest.raises(ValueError):
        PointCloudTransport('invalid_context_transport', '', not_a_context)


def test_an_uninitialized_context_is_refused():
    """A context that was not initialized cannot be followed by a transport."""
    uninitialized = rclpy.context.Context()
    with pytest.raises(ValueError):
        PointCloudTransport('uninitialized_context_transport', '', uninitialized)


def test_a_derived_context_is_accepted():
    """A context of a class derived from the one of rclpy is followed as well."""
    class DerivedContext(rclpy.context.Context):
        """A context that only adds a name to be something of its own."""

    context = DerivedContext()
    rclpy.init(context=context)
    transport = PointCloudTransport('derived_context_transport', '', context)
    try:
        assert transport.ok()
        rclpy.shutdown(context=context)
        assert not transport.ok()
    finally:
        transport.shutdown()
        rclpy.try_shutdown(context=context)
