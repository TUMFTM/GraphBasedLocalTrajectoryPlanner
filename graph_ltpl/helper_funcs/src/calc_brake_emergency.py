import numpy as np
import trajectory_planning_helpers as tph

# vehicle parameters
VEH_MASS = 1160.0
VEH_DRAGCOEFF = 0.854


def calc_brake_emergency(traj: np.ndarray,
                         loc_gg: np.ndarray) -> np.ndarray:
    """
    Calculates a simple emergency profile (brake to stop) for a given regular trajectory.

    :param traj:        trajectory with the columns (s, x, y, heading, curv, vel, acc)
    :param loc_gg:      local gg scaling along the path.
    :returns:
        * **traj_em** - emergency trajectory (braking as much as feasible on the path of 'traj' - same format as 'traj')

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        17.04.2020

    """

    # calculate element length
    el_lengths = np.diff(traj[:, 0])

    # calculate brake-vel-profile
    v_brake = tph.calc_vel_profile_brake.calc_vel_profile_brake(kappa=traj[:, 4],
                                                                el_lengths=el_lengths,
                                                                v_start=traj[0, 5],
                                                                drag_coeff=VEH_DRAGCOEFF,
                                                                m_veh=VEH_MASS,
                                                                loc_gg=loc_gg)

    # calculate matching acceleration profile
    idx_em = len(v_brake)
    a_brake = tph.calc_ax_profile.calc_ax_profile(vx_profile=v_brake,
                                                  el_lengths=el_lengths[:idx_em],
                                                  eq_length_output=True)

    # assemble emergency trajectory
    traj_em = np.column_stack((traj[:idx_em, 0:5], v_brake, a_brake))

    return traj_em


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
