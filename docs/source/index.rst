.. Agent Simulation documentation master file, created by
   sphinx-quickstart on Thu May 28 14:28:26 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Graph-Based Local Trajectory Planner Documentation
==================================================

.. image:: /figures/Title.png
  :alt: Trajectory Planner Title Image


The graph-based local trajectory planner is python-based and comes with open interfaces as well as debug, visualization
and development tools. The local planner is designed in a way to return an action set (e.g. keep straight, pass left,
pass right), where each action is the globally cost optimal solution for that task. If any of the action primitives is
not feasible, it is not returned in the set. That way, one can either select available actions based on a priority list
(e.g. try to pass if possible) or use an own dedicated behaviour planner.

The planner was used on a real race vehicle during the Roborace Season Alpha and achieved speeds above 200kph.
A video of the performance at the Monteblanco track can be found `here <https://youtu.be/-vqQBuTQhQw>`_.

.. warning::
   This software is provided *as-is* and has not been subject to a certified safety validation. Autonomous Driving is a
   highly complex and dangerous task. In case you plan to use this software on a vehicle, it is by all means required
   that you assess the overall safety of your project as a whole. By no means is this software a replacement for a valid
   safety-concept. See the license for more details.

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   start/overview.rst
   start/installation.rst
   start/launching.rst

.. toctree::
   :maxdepth: 2
   :caption: Software Explanation:
   :glob:

   software/inputs.rst
   software/config.rst
   software/basics.rst
   software/logs.rst
   software/devtools.rst

.. note:: Further details about the actual implementation and the purpose of individual functions can be found in the
    :doc:`graph_ltpl/modules`.


.. toctree::
   :maxdepth: 2
   :caption: Code Documentaion:

   graph_ltpl/modules.rst


Contributions
=============
[1] T. Stahl, A. Wischnewski, J. Betz, and M. Lienkamp,
“Multilayer Graph-Based Trajectory Planning for Race Vehicles in Dynamic Scenarios,”
in 2019 IEEE Intelligent Transportation Systems Conference (ITSC), Oct. 2019, pp. 3149–3154.
`(view pre-print) <https://arxiv.org/pdf/2005.08664>`_

If you find our work useful in your research, please consider citing:

.. code-block:: latex

   @inproceedings{stahl2019,
     title = {Multilayer Graph-Based Trajectory Planning for Race Vehicles in Dynamic Scenarios},
     booktitle = {2019 IEEE Intelligent Transportation Systems Conference (ITSC)},
     author = {Stahl, Tim and Wischnewski, Alexander and Betz, Johannes and Lienkamp, Markus},
     year = {2019},
     pages = {3149--3154}
   }

Contact Information
===================

:Email: `tim.stahl@tum.de <tim.stahl@tum.de>`_
