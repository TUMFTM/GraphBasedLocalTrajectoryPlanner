================================
Inputs to the Trajectory Planner
================================

All relevant input files can be found in the '`input`'-folder, located in the root-directory. All the files / folders
are explained in the following.

traj_ltpl_cl
========================
This folder holds a set of global race lines with corresponding map information. The files hold comments in the first
two lines, a semicolon separated header in the third line and corresponding data in the following lines. The file must
hold the following columns with the headers listed:

* x_ref_m - x coordinates of the reference line (e.g. center line of the track)
* y_ref_m - y coordinates of the reference line (e.g. center line of the track)
* width_right_m - width of the track measured from the corresponding point on the reference line to the right (along
  the normal vector)
* width_left_m - width of the track measured from the corresponding point on the reference line to the left (along
  the normal vector)
* x_normvec_m - x coordinate of the normalized normal vector based on the corresponding point of the reference line
* y_normvec_m - y coordinate of the normalized normal vector based on the corresponding point of the reference line
* alpha_m - location of the point building the race line on the normal vector of a point belonging to the corresponding
  reference line
* s_racetraj_m - travelled distance along the race line (starting at 0.0 m for the first point)
* psi_racetraj_rad - heading of the race line point (north = 0.0)
* kappa_racetraj_radpm - curvature along the race line
* vx_racetraj_mps - globally optimal velocity profile along the race line
* ax_racetraj_mps2 - acceleration profile matching the velocity profile

It should be noted, that all trajectory files should start with 'traj_ltpl_cl_' in order to work with the provided
sample files. That way, the last part can be used to specify the track to be driven in the 'driving_task.ini'.

Further race lines and track information can be generated with an global race trajectory optimization algorithm. The
open-source version of the global planner providing matching files for this local planner is hosted on
`GitHub <https://github.com/TUMFTM/global_racetrajectory_optimization>`_. (In order to generate the appropriate files,
read the Readme and enable the "LTPL Trajectory" output.

veh_dyn_info
========================
The dynamical behavior of the vehicle for the initially generated velocity profile can be adjusted with the files in the
'`params/veh_dyn_info`'-folder. The '`ax_max_machines.csv`'-file describes the acceleration resources of the motor at
certain velocity thresholds (values in between are interpolated linearly).

.. note::
    The 'ax_max_machines.csv' can be imported with a function of the 'trajectory-planning-helpers' package.

    .. code-block:: python

        import trajectory_planning_helpers as tph

        ax_max = tph.import_veh_dyn_info.\
            import_veh_dyn_info(ax_max_machines_import_path="/path/to/ax_max_machines.csv")[1]

    The imported matrix can then be provided to the 'calc_vel_profile()' function of the Graph_LTPL class.
