===================================
Launching the Planner
===================================

In order to use the planner, integrate the graph_ltpl package in your code. In order to get started, we provide two
example scripts. These scripts demonstrate the code integration and allow to test the planners features.

Minimal Example
===============
A minimal example of the local trajectory planner is parametrized in the '`main_min_example.py`' script in the root
directory. Within this example, the planner is executed without any interfaces (e.g. object list) and no configured
logging. By default a live-visualization is shown (note: the live-visualization slows down the execution drastically).
Launch the code with the following command:

.. code-block:: bash

    python3 main_min_example.py

.. note:: When a certain track configuration is executed the first time, a offline graph is generated first. This will
    take some time (progress bar is displayed). On completion, a plot of the generated graph with all its edges is
    displayed (in case the visualization is enabled). After closing the figure with the offline graph, a live
    visualization is launched and the vehicle starts driving.

Standard Example
================
A more comprehensive example of the local trajectory planner is given in the '`main_std_example.py`' script (also in the
root directory). Within this example, the planner is executed with basic interfaces:

* Dummy object list - another vehicle driving with reduced speed along the racing-line (NOTE: here executed without any
  prediction of the object-vehicle's motion)
* Dummy blocked zone - a small region of the track is blocked for the ego vehicle (useful to block certain regions, e.g.
  pit lane or dirty track)
* Logging to file - the environment and planned trajectories of every time-stamp are logged to a file, which can be
  visualized with the interactive log-viewer afterwards
* Prioritized action selection - instead of a behavior planner, in this example any 'pass' action is executed, when
  available

By default a live-visualization is shown (note: the live-visualization slows down the execution drastically).
Launch the code with the following command:

.. code-block:: bash

    python3 main_std_example.py

.. note:: When a certain track configuration is executed the first time, a offline graph is generated first. This will
    take some time (progress bar is displayed). On completion, a plot of the generated graph with all its edges is
    displayed (in case the visualization is enabled). After closing the figure with the offline graph, a live
    visualization is launched and the vehicle starts driving.


Further Steps
=============
A description for usage of the planner class in your project is given in :doc:`../software/basics`.
Furthermore, the parameterization, development tools and the log visualizer is tackled in that Chapter.

