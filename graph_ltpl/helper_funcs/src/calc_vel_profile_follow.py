import numpy as np
import logging
import time
import math

# custom modules
import graph_ltpl

# custom libraries
import trajectory_planning_helpers as tph


def clamp(n, minn, maxn):
    """
    Clamp a value 'n' within the range 'minn' and 'maxn'.

    :param n:               value to be clamped
    :param minn:            lower bound of clamp
    :param maxn:            upper bound of clamp
    :returns:
        * **clamped_val** - value within the range of 'minn' and 'maxn'.

    """

    return min(max(n, minn), maxn)


def get_control_vel(control_params: dict,
                    obj_dist: float,
                    control_d: float,
                    v_obj: float,
                    v_ego: float,
                    control_type: str = 'PD'):
    """
    Calculate a desired velocity in order to follow an object vehicle with a desired distance.

    :param control_params:      dict of control parameters, depending on the chosen approach ('control_type'):

                                * control_type=='PD' - provide parameters:
                                    * 'k_d': differential control coefficient (weighting velocity mismatch)
                                    * 'k_p': proportional control coefficient (weighting offset from control distance)
                                * control_type=='PDtan' - provide parameters:
                                    * 'k_d': differential control coefficient (weighting velocity mismatch)
                                    * 'k_p': proportional control coefficient (weighting offset from control distance,
                                      output axis of tan)
                                    * 'tan_w': control error distance at which the maximum value of the tangens is
                                      reached (scales tangens along input axis)
    :param obj_dist:            current distance to object [in m]
    :param control_d:           desired control distance (to object) [in m]
    :param v_obj:               object velocity [in mps]
    :param v_ego:               ego velocity [in mps]
    :param control_type:        string specifying control type ('PD': PD flavoured control law, 'PDtan': PD flavoured
                                control law with tan-weighting around control point.)
    :returns:
        * **v_control** -       desired velocity (in order to approach control_d smoothly)

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        14.02.2019

    """

    if control_type == 'PD':
        v_control = (v_obj - control_params['k_p']
                     * (control_d - obj_dist) + control_params['k_d'] * (v_obj - v_ego))
    elif control_type == 'PDtan':
        v_control = (v_obj - math.tan(clamp((control_d - obj_dist) * math.pi / 2 * 1 / control_params['tan_w'],
                                            -math.pi / 2 + 1e-5, math.pi / 2 - 1e-5)) * control_params['k_p']
                     + control_params['k_d'] * (v_obj - v_ego))
    else:
        raise ValueError('Unsupported control type "' + control_type + '"!')

    return v_control


def calc_vel_profile_follow(ax_max_machines: np.ndarray,
                            dyn_model_exp: float,
                            drag_coeff: float,
                            m_veh: float,
                            kappa: np.ndarray,
                            el_lengths: np.ndarray,
                            loc_gg: np.ndarray,
                            v_start: float,
                            v_ego: float,
                            v_obj: float,
                            v_max: float,
                            safety_d: float,
                            veh_length: float,
                            obj_dist: float,
                            obj_pos: list,
                            glob_rl: np.ndarray,
                            control_type: str,
                            control_params: dict) -> tuple:
    """
    Calculates the velocity profile for a given trajectory, an target index (within the trajectory) as well as the
    velocity of the ego and the target vehicle.

    :param ax_max_machines:         velocity dependent maximum acceleration allowed by the machine
    :param dyn_model_exp:           dynamic model exponent (in range of [1, 2] specifying shape of friction range)
    :param drag_coeff:              drag coefficient of the ego-vehicle
    :param m_veh:                   mass of the ego-vehicle [in kg]
    :param kappa:                   course of curvature in [1/m]
    :param el_lengths:              spatial distance between the curvature values [in m]
    :param loc_gg:                  local gg diagram (lateral and longitudinal allowed acceleration per curvature value)
    :param v_start:                 start velocity (planned; at current position) [in mps]
    :param v_ego:                   estimated velocity (actual; at current position) [in mps]
    :param v_obj:                   velocity of target object [in mps]
    :param v_max:                   maximum velocity allowed [in mps]
    :param safety_d:                safety distance to be maintained [in m]
    :param veh_length:              length of vehicle (assumes both vehicles [target and ego] to share same dims) [in m]
    :param obj_dist:                distance to target vehicle [in m]
    :param obj_pos:                 position of closest vehicle (x and y coordinate)
    :param glob_rl:                 global race line used for assumed break distance
    :param control_type:            string specifying control type ('PD': PD flavoured control law, 'PDtan': PD
                                    flavoured control law with tan-weighting around control point.)
    :param control_params:          control parameters (setpoint, coefficients, ...)
    :returns:
        * **vx_final** -            velocity course for the trajectory in order to reach the desired control distance
        * **too_close** -           flag holding 'True' if vehicle is within the safety distance to the other vehicle
        * **vel_bound_fulfilled** - flag holding 'True' if velocity profile is not feasible for the given constraints

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        14.02.2019

    """

    # User Input -------------------------------------------------------------------------------------------------------
    # ggv diagram used to calculate the brake distance of opponent vehicles (rather estimate to high, than too low)
    ggv = np.atleast_2d([100.0, 14.0, 14.0])

    # ------------------------------------------------------------------------------------------------------------------

    vel_bound_fulfilled = True

    # Determine control distance (set point for the controller)
    control_d = control_params['c_p'] * safety_d + veh_length

    # Add vehicle length to safety distance (after control distances are determined)
    safety_d = safety_d + veh_length

    # check if safety distance is violated
    too_close = False
    if (obj_dist - safety_d) < 0:
        too_close = True

    # calculate brake velocity profile for ego vehicle on local trajectory
    v_profile_ego_brake = tph.calc_vel_profile_brake. \
        calc_vel_profile_brake(loc_gg=loc_gg,
                               kappa=kappa,
                               el_lengths=el_lengths[:kappa.shape[0] - 1],
                               v_start=v_start,
                               dyn_model_exp=dyn_model_exp,
                               drag_coeff=drag_coeff,
                               m_veh=m_veh)

    # find braking distance from start to end point of ego
    id_brake = 0
    while id_brake < len(kappa) and v_profile_ego_brake[id_brake] > 0.1:
        id_brake += 1

    ego_stop_dist = np.sum(el_lengths[0:id_brake])

    # calculate element lengths for global trajectory
    glob_rl = np.column_stack((glob_rl[:-1], np.diff(glob_rl[:, 0])))

    # matching opponent car to global race trajectory
    s_opp, idxs_tmp = graph_ltpl.helper_funcs.src.get_s_coord.get_s_coord(ref_line=glob_rl[:, 1:3],
                                                                          pos=tuple(obj_pos),
                                                                          s_array=glob_rl[:, 0],
                                                                          closed=True)
    idx_s_opp = idxs_tmp[0]

    # reorder global trajectory such that idx_s_opp is the starting point
    glob_rl_rolled = np.roll(glob_rl, glob_rl.shape[0] - idx_s_opp, axis=0)

    # use smaller start velocity, either velocity of opponent or velocity of global trajectory
    vel_start = min(v_obj, glob_rl_rolled[0, 4])

    # calculate brake velocity profile for opponent on global raceline
    v_profile_opp_brake = tph.calc_vel_profile_brake. \
        calc_vel_profile_brake(ggv=ggv,
                               kappa=glob_rl_rolled[:, 3],
                               el_lengths=glob_rl_rolled[:-1, 5],
                               v_start=vel_start,
                               dyn_model_exp=dyn_model_exp,
                               drag_coeff=drag_coeff,
                               m_veh=m_veh)

    # find braking distance from start to end point of opponent
    id_brake = 0
    while id_brake < len(v_profile_opp_brake) and v_profile_opp_brake[id_brake] > 0.1:
        id_brake += 1

    opp_stop_dist = np.sum(glob_rl_rolled[0:id_brake, 5])

    # -- GET INDEXES OF CHARACTERISTIC POSITIONS -----------------------------------------------------------------------
    # calculate s coordinates along given path
    s = np.concatenate(([0], np.cumsum(el_lengths[:-1])))

    stop_idx = 0
    s_stop = obj_dist - safety_d + opp_stop_dist

    while stop_idx < len(s) - 1 and s[stop_idx] < s_stop:
        stop_idx += 1

    v_end = 0.0
    if s_stop > s[-1]:
        s_loctraj_ends = opp_stop_dist - (s_stop - s[-1])

        idx = 0
        s_summed = 0.0
        while s_summed < s_loctraj_ends and idx < glob_rl_rolled.shape[0]:
            s_summed += glob_rl_rolled[idx, 5]
            idx += 1

        v_end = glob_rl_rolled[idx, 4]

    # -- CALCULATE CONTROL VELOCITY (based on a simple control mechanism) ----------------------------------------------
    # d' = v_obj - v_control
    #    = v_obj - (v_obj - K * (c * safety_d - actual_d)
    # d' = K * (c * safety_d - actual_d)
    # ds = K * (c * safety_d - actual_d)
    # d(s + K) = k * c * safety_d
    # d / (c * safety_d) = K / (s + K) --> 1 / (1/K * s + 1) --> T = 1/K

    # calculate desired control velocity
    v_control = get_control_vel(control_params=control_params,
                                obj_dist=obj_dist,
                                control_d=control_d,
                                v_obj=v_obj,
                                v_ego=v_ego,
                                control_type=control_type)
    v_control = max(v_control, 0.0)
    v_control = min(v_control, v_max)

    # log controller parameters
    if "follow_mode_logger" in logging.Logger.manager.loggerDict.keys():
        clog = (str(time.time()) + ";" + str(obj_dist) + ";" + str(control_d) + ";" + str(v_control) + ";"
                + str(v_obj) + ";" + str(v_ego))
        logging.getLogger("follow_mode_logger").info(clog)

    if ego_stop_dist < s_stop:
        # if not possible to stop within safety requirements
        # -- SEGMENT 1 (if faster than control velocity) ---------------------------------------------------------------
        if v_start > v_control and stop_idx >= 2:
            vx_decel = v_profile_ego_brake

            # cut profile at position where new maximum vel is reached
            idx_c = min(int(np.argmax(vx_decel <= v_control)), stop_idx)
            if idx_c == 0:
                idx_c = stop_idx
            vx_decel = vx_decel[:(idx_c + 1)]
            vx_control_start = vx_decel[-1]
        else:
            if not stop_idx >= 2:
                vel_bound_fulfilled = False
            idx_c = 0
            vx_decel = []
            vx_control_start = v_start

        # -- SEGMENT 2 (normal velocity planning within control velocity) ----------------------------------------------
        if (stop_idx - idx_c) > 0:
            vx_control = tph.calc_vel_profile. \
                calc_vel_profile(loc_gg=loc_gg[idx_c:(stop_idx + 1)],
                                 ax_max_machines=ax_max_machines,
                                 v_max=v_control,
                                 kappa=kappa[idx_c:(stop_idx + 1)],
                                 el_lengths=el_lengths[idx_c:stop_idx],
                                 v_start=vx_control_start,
                                 v_end=v_end,
                                 dyn_model_exp=dyn_model_exp,
                                 drag_coeff=drag_coeff,
                                 m_veh=m_veh,
                                 closed=False)
            if np.abs(vx_control[0] - vx_control_start) > 1.0:
                vel_bound_fulfilled = False

        elif (stop_idx - idx_c) == 0:
            vx_control = [vx_control_start]
        else:
            vx_control = []

        # concatenate velocity profiles (remove overlapping starting point if there is any)
        vx_profile = np.concatenate((vx_decel[:-1], vx_control, [0.0] * (len(kappa) - stop_idx - 1)))

        if np.abs(vx_profile[0] - v_start) > 1.0:
            vel_bound_fulfilled = False
    else:
        vx_profile = v_profile_ego_brake

    # CALCULATE COMPLETE VELOCITY PROFILE ------------------------------------------------------------------------------
    vx_profile_compl = tph.calc_vel_profile. \
        calc_vel_profile(loc_gg=loc_gg,
                         ax_max_machines=ax_max_machines,
                         v_max=v_max,
                         kappa=kappa,
                         el_lengths=el_lengths[:-1],
                         v_start=v_start,
                         dyn_model_exp=dyn_model_exp,
                         drag_coeff=drag_coeff,
                         m_veh=m_veh,
                         closed=False)

    # intersect complete velocity profile with previous one
    vx_final = np.minimum(vx_profile, vx_profile_compl)
    # vx_final = vx_profile

    return vx_final, too_close, vel_bound_fulfilled


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
