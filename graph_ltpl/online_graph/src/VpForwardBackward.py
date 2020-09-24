import numpy as np
import logging

# custom modules
import graph_ltpl

# custom packages
import trajectory_planning_helpers as tph


class VpForwardBackward(object):
    """
    Class providing interfaces and relevant calculations for the forward-backward velocity planner.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        13.01.2020
    """

    def __init__(self,
                 dyn_model_exp: float,
                 drag_coeff: float,
                 m_veh: float,
                 len_veh: float,
                 follow_control_type: str,
                 follow_control_params: dict,
                 glob_rl: np.ndarray) -> None:
        """
        :param dyn_model_exp:          vehicle dynamics model exponent (range [1.0, 2.0])
        :param drag_coeff:             drag coefficient -> 0.5 * rho_air * c_w_A
        :param m_veh:                  vehicle mass
        :param len_veh:                vehicle length
        :param follow_control_type:    string specifying the controller to be used for follow-mode set point generation
        :param follow_control_params:  dict of controller parameters for the follow-mode set point generation
        :param glob_rl:                global race line

        """

        # init logger handle
        self.__log = logging.getLogger("local_trajectory_logger")

        self.__follow_control_type = follow_control_type
        self.__follow_control_params = follow_control_params

        # parameters for velocity profile
        self.__dyn_model_exp = dyn_model_exp
        self.__drag_coeff = drag_coeff
        self.__m_veh = m_veh
        self.__len_veh = len_veh

        self.__glob_rl_clsd = glob_rl

        # class member variables
        self.__vel_max = None
        self.__gg_scale = None
        self.__ax_max_machines = None

        self.__old_gg_scale = None

    def __del__(self) -> None:
        pass

    def update_dyn_parameters(self,
                              vel_max: float,
                              gg_scale: float,
                              ax_max_machines: np.ndarray) -> None:
        """
        Update dynamic vehicle parameters (updated once per iteration).

        :param vel_max:         maximum velocity allowed by the behavior planner
        :param gg_scale:        gg-scale in range [0.0, 1.0] applied to [ax_max, ay_max]
        :param ax_max_machines: velocity dependent maximum acceleration allowed by the machine

        """

        # store the previous gg-scale (to be used in case of recursive infeasibility)
        if self.__old_gg_scale is None:
            self.__old_gg_scale = gg_scale

        self.__vel_max = vel_max
        self.__gg_scale = gg_scale
        self.__ax_max_machines = ax_max_machines

    def check_brake_prefix(self,
                           vel_plan: float,
                           vel_course: np.ndarray,
                           kappa: np.ndarray,
                           el_lengths: np.ndarray,
                           loc_gg: np.ndarray) -> tuple:
        """
        Check if braking to a new maximum velocity is necessary. If so, calculate according brake profile with previous
        acceleration limits.

        :param vel_plan:        velocity of the ego vehicle according to the last plan
        :param vel_course:      velocity course of the last trajectory, which should stay constant due to delay
        :param kappa:           curvature profile of given trajectory in rad/m.
        :param el_lengths:      element lengths (distances between coordinates) of given trajectory.
        :param loc_gg:          allowed local gg values along path
        :returns:
            * **vx_prefix** -   velocity course required for decel. to new maximum velocity (lower than current speed),
              this also includes the constant velocity part
            * **pref_idx** -    index in the velocity profile where deceleration is completed (excl. constant vel. part)
            * **vel_start** -   start velocity for adjacent velocity profile

        """

        if vel_plan > (self.__vel_max + 0.1):
            self.__log.info("Applying deceleration in order to break to new v_max!")

            # calculate brake profile starting at current velocity (using old gg-scale)
            # increase g limits in order to cope with numerical issues in pure forward solver
            gg_brake = loc_gg * self.__old_gg_scale
            vx_decel = tph.calc_vel_profile_brake. \
                calc_vel_profile_brake(loc_gg=gg_brake,
                                       kappa=kappa,
                                       el_lengths=el_lengths,
                                       v_start=vel_plan,
                                       dyn_model_exp=self.__dyn_model_exp,
                                       drag_coeff=self.__drag_coeff,
                                       m_veh=self.__m_veh)

            # cut profile at position where new maximum vel is reached
            idx = np.argmax(vx_decel <= self.__vel_max)
            if idx == 0:
                idx = len(vx_decel) - 1

            # velocity profile based on delay compensation and deceleration
            vx_prefix = np.concatenate((vel_course, vx_decel[:idx]))
            pref_idx = idx
            vel_start = vx_decel[idx]
        else:  # -> only delay
            vx_prefix = vel_course
            pref_idx = 0
            vel_start = vel_plan
            self.__old_gg_scale = self.__gg_scale

        return vx_prefix, pref_idx, vel_start

    def calc_vel_profile_follow(self,
                                kappa: np.ndarray,
                                el_lengths: np.ndarray,
                                loc_gg: np.array,
                                v_start: float,
                                v_ego: float,
                                v_obj: float,
                                safety_d: float,
                                obj_dist: float,
                                obj_pos: list) -> tuple:
        """
        Calculates the velocity profile for a given trajectory, an target index (within the trajectory) as well as the
        velocity of the ego and the target vehicle.

        :param kappa:              course of curvature in [1/m]
        :param el_lengths:         spatial distance between the curvature values [in m]
        :param loc_gg:             local gg diagram (lateral and longitudinal allowed acceleration per curvature value)
        :param v_start:            start velocity (planned; at current position) [in mps]
        :param v_ego:              estimated velocity (actual; at current position) [in mps]
        :param v_obj:              velocity of target object [in mps]
        :param safety_d:           safety distance to be maintained [in m]
        :param obj_dist:           distance to target vehicle [in m]
        :param obj_pos:            position of closest vehicle (x and y coordinate)
        :returns:
            * **vx** -             calculated velocity profile (same amount of data points as 'kappa' holds)
            * **too_close** -      flag holding 'True' if vehicle is within the safety distance to the other vehicle
            * **vel_bound** -      flag holding 'True' if velocity profile is not feasible for the given constraints

        """

        # calculate follow profile with fb solver
        vx, too_close, vel_bound = graph_ltpl.helper_funcs.src.calc_vel_profile_follow. \
            calc_vel_profile_follow(loc_gg=loc_gg * self.__gg_scale,
                                    ax_max_machines=self.__ax_max_machines,
                                    dyn_model_exp=self.__dyn_model_exp,
                                    drag_coeff=self.__drag_coeff,
                                    m_veh=self.__m_veh,
                                    kappa=kappa,
                                    el_lengths=el_lengths,
                                    v_start=v_start,
                                    v_ego=v_ego,
                                    v_obj=v_obj,
                                    v_max=self.__vel_max,
                                    safety_d=safety_d,
                                    veh_length=self.__len_veh,
                                    obj_dist=obj_dist,
                                    obj_pos=obj_pos,
                                    glob_rl=self.__glob_rl_clsd,
                                    control_type=self.__follow_control_type,
                                    control_params=self.__follow_control_params)

        return vx, too_close, vel_bound

    def calc_vel_profile(self,
                         kappa: np.ndarray,
                         el_lengths: np.ndarray,
                         loc_gg: np.ndarray,
                         v_start: float,
                         v_end: float) -> np.ndarray:
        """
        Calculates a velocity profile using the tire and motor limits.

        :param kappa:              course of curvature in [1/m]
        :param el_lengths:         spatial distance between the curvature values [in m]
        :param loc_gg:             local gg diagram (lateral and longitudinal allowed acceleration per curvature value)
        :param v_start:            start velocity (planned; at current position) [in mps]
        :param v_end:              end velocity (planned; at end of trajectory) [in mps]
        :returns:
            * **vx** -             calculated velocity profile (same amount of data points as 'kappa' holds)

        """

        vx = tph.calc_vel_profile.calc_vel_profile(
            loc_gg=loc_gg * self.__gg_scale,
            ax_max_machines=self.__ax_max_machines,
            v_max=self.__vel_max,
            kappa=kappa,
            el_lengths=el_lengths,
            v_start=v_start,
            v_end=v_end,
            dyn_model_exp=self.__dyn_model_exp,
            drag_coeff=self.__drag_coeff,
            m_veh=self.__m_veh,
            closed=False
        )

        return vx

    def calc_vel_brake_em(self,
                          kappa: np.ndarray,
                          el_lengths: np.ndarray,
                          loc_gg: np.array,
                          v_start: float):
        """
        Calculates an emergency brake profile using the tire and motor limits. (e.g. to be applied in case of recursive
        infeasibility) --> No gg-scale used here

        :param kappa:              course of curvature in [1/m]
        :param el_lengths:         spatial distance between the curvature values [in m]
        :param loc_gg:             local gg diagram (lateral and longitudinal allowed acceleration per curvature value)
        :param v_start:            start velocity (planned; at current position) [in mps]
        :returns:
            * **vx** -             calculated velocity profile (same amount of data points as 'kappa' holds)

        """

        vx = tph.calc_vel_profile_brake.calc_vel_profile_brake(loc_gg=loc_gg,
                                                               kappa=kappa,
                                                               el_lengths=el_lengths,
                                                               v_start=v_start,
                                                               dyn_model_exp=self.__dyn_model_exp,
                                                               drag_coeff=self.__drag_coeff,
                                                               m_veh=self.__m_veh)

        return vx
