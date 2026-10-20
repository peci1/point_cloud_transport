API Reference
=============

Compiled Classes
----------------

The following classes are provided by the pybind11 bindings to
``point_cloud_transport``.  Their docstrings live in the compiled C++ module and
mirror the C++ API — see the
`point_cloud_transport <https://docs.ros.org/en/rolling/p/point_cloud_transport/>`__
documentation for a full description.

PointCloudTransport
~~~~~~~~~~~~~~~~~~~

Main entry point.  Owns the plugin loaders and exposes ``advertise()`` and
``subscribe()`` methods that return :class:`Publisher` and :class:`Subscriber`
handles.  It runs on a node of its own, which is serviced by a thread the
transport owns and which is created with the ROS arguments and the domain id that
the constructor is given.

.. code-block:: python

   from point_cloud_transport_py import PointCloudTransport
   pct = PointCloudTransport('my_transport')
   pub = pct.advertise('points', 10)
   sub = pct.subscribe('points', 10, callback)

The constructor takes the name of the node to create, an optional path to a
parameters file, an optional ``rclpy.context.Context``, and optional ROS arguments
and domain id for the node:

.. code-block:: python

   import sys

   import rclpy
   from point_cloud_transport_py import PointCloudTransport

   context = rclpy.context.Context()
   rclpy.init(context=context)
   with PointCloudTransport('my_transport', context=context, argv=sys.argv) as pct:
       pub = pct.advertise('points', 10)

The node runs in an ``rclcpp`` context of its own, which the transports created
with the same options share, so that a transport can sit next to an ``rclpy``
program without disturbing it.  The context of ``rclpy`` itself cannot be used:
it hands out the address of its ``rcl_context`` and nothing more, and
``rclcpp::Context`` makes one itself in ``init()`` and finalizes it again on its
own, so there is no ``rcl_context`` for the two to share.  What a transport does
take from the context it is given is a copy of the options that context was
initialized with, so the enclave of the process and whatever else the middleware
was told about it apply to the node as well, the parameter files that context was
told about, the domain id the node talks on, and the moment it is released: the
transport cleans itself up as soon as that context is shut down.  The argument is
checked against ``rclpy.context.Context``, on which this package depends, so only
a context of rclpy is accepted.  ``ok()`` tells whether a transport can still be
used, and ``name()`` gives the name of its node.

``argv`` are the command line arguments to initialize the context of the node
with, the way ``rclcpp`` takes them from a program: the arguments that follow
``--ros-args`` carry the remappings, the parameters and the enclave of a node.
``rcl`` keeps no copy of the arguments a context was initialized with, and
everything it made of them is in a form that no new context can be initialized
with, so a transport that is to follow the arguments of a program has to be given
them itself, and ``sys.argv`` is what they usually come from.  The enclave and the
paths of the parameter files are the two things that can be read back from a given
context, so those two are asked for again; the remappings, the parameter overrides
and the log levels a context was given stay with that context.  Whatever is in
``argv`` is put behind what was read back, so it has the last word.  Left at
``None`` a node gets no ROS arguments of its own.  ``domain_id`` says which domain
the node talks on and has to agree with a context that is given; left at ``None``
it is the one of that context, or the one of the ``ROS_DOMAIN_ID`` environment
variable.

What is left of the ``InitOptions`` of ``rclcpp`` is deliberately not exposed and
not taken over from a given context: the allocator is for ``rclcpp`` and ``rcl`` to
take, the logging of a process is set up by ``rclpy`` once for all, so a
``--log-level`` reaches no node, and Ctrl-C belongs to the interpreter, so no
signal handler is installed for a transport and nothing shuts its context down on
a signal.

``shutdown()`` ends the thread servicing the node and destroys the node.  It is
also what the destructor and leaving a ``with`` block do, and calling it more
than once is allowed.  A transport that was shut down refuses to advertise.

Publisher
~~~~~~~~~

Handle returned by :meth:`PointCloudTransport.advertise`.  Use
``publish(msg)`` to send a ``sensor_msgs/PointCloud2`` on every configured
transport; use ``shutdown()`` to release the underlying advertisements.

Subscriber
~~~~~~~~~~

Handle returned by :meth:`PointCloudTransport.subscribe`.  Calls the user
callback with a ``sensor_msgs/PointCloud2`` each time a message is received.
Use ``shutdown()`` to unsubscribe.

PointCloudCodec
~~~~~~~~~~~~~~~

Node-free encode/decode facade.  Provides ``getLoadableTransports()``,
``getEncoderByName()``, ``getDecoderByName()``, ``encode()``, and ``decode()``
methods that operate on serialized messages.

common module
-------------

The :mod:`point_cloud_transport_py.common` module provides serialization helpers
and the :class:`~point_cloud_transport_py.common.TransportInfo` dataclass:

- :class:`point_cloud_transport_py.common.TransportInfo`
- :func:`point_cloud_transport_py.common.stringToPointCloud2`
- :func:`point_cloud_transport_py.common.pointCloud2ToString`
- :func:`point_cloud_transport_py.common.stringToMsgType`

Full module documentation: :doc:`/point_cloud_transport_py.common`.

Package overview: :doc:`/point_cloud_transport_py`.
