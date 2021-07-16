========================
Configuration
========================

All relevant configuration files can be found in the '`params`'-folder, located in the root-directory. All the files are
explained in the following.

driving_task.ini
========================
The driving task file holds information about the task to be executed. Currently, it solely holds the track to be loaded
and driven. Further information, like a maximum velocity may be added by the user.


ltpl_config_offline.ini
========================
The offline configuration file holds parameters primarily specifying the offline generation of the graph. Thereby, the
discretization of the lattice as well as offline cost weighting can be adjusted. All parameters are described in detail
in the config file.

ltpl_config_online.ini
========================
The online configuration file holds parameters specifying the online planning of with the graph. As with the offline
config file, all parameters are described in detail in the config file.

An important parameter in the config file is 'vp_type', which specifies the executed velocity planner. Currently to variants
are integrated into the planner. 'fb' is a forward-backward planner, which is enabled by default and fully integrated
into the stack. 'sqp' is an optimization based approach and must be installed separatly (including parameter and input files).

.. note:: Details on the SQP velocity planer (incl. the parameter files), as well as required packages can be found in
    the corresponding repository ('https://github.com/TUMFTM/velocity_optimization').
