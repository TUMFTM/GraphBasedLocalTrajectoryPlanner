===================================
Development Tools
===================================
In order to ease trajectory planning development, we provide some useful tools in the 'graph_ltpl/testing_tools'
folder. The intent of these tools is to test the planner without the need to always launch the whole software stack
(e.g. a dedicated perception and controller module). In that way, these scripts provide a rudimentary / simplified
functionality of an object list ('objectlist_dummy.py') and a controller that ideally tracks the planned trajectory
('vdc_dummy.py').

objectlist_dummy.py
===================
The dummy object list serves the purpose of providing a (simple) object list. By default, a single object vehicle is
driving with reduced speed along the race line.

The object list dummy can be executed in two ways:

* Retrieve the object list via function call. The class keeps track of the passed time between the function calls and
  moves the object vehicle accordingly. (This variant is implemented in the 'main_std_example.py' script)
* Retrieve the object list via ZMQ communication. The object list dummy is launched in a separate thread and publishes
  the object list via ZMQ in a specified frequency. The list can then be received in the planner with a matching ZMQ
  receiver. This variant corresponds more to the situation in the vehicle.

vdc_dummy.py
============
The vehicle dynamics controller (VDC) dummy emulates an ideal controller that tracks the planned trajectory perfectly.
The script can be called iteratively and requests the last planned trajectory, last position estimate as well as the
passed time since the last call. The script then travels the specified iteration time along the trajectory and returns
the resulting position and velocity defined by the planned trajectory.

.. note:: Details on the usage / parameterization can be found in the :doc:`../../software_imp/modules` ('testing_tools'
    package). Furthermore, an exemplary usage of both of the tools is implemented in the 'main_std_example.py' script.
