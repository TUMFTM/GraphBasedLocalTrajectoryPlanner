import numpy as np
import os
import logging

# custom packages
try:
    import velocity_optimization as vo
    import_failed = False
except ImportError:
    import_failed = True


def get_import_failure_status() -> bool:
    return import_failed


class VpSQP(object):
    """
    Class providing interfaces and relevant calculations for the SQP velocity planner.

    :Authors:
        * Thomas Herrmann <thomas.herrmann@tum.de>
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        25.08.2020
    """

    def __init__(self,
                 nmbr_export_points: int,
                 stepsize_approx: float,
                 veh_turn: float,
                 glob_rl: np.ndarray,
                 delaycomp: float) -> None:
        """
        :param nmbr_export_points:     number of exported bounds (longer trajectories will be cut to this length)
        :param stepsize_approx:        approximated step-size for all generated splines [in m]
        :param veh_turn:               vehicle turn radius [in m]
        :param glob_rl:                global race line
        :param delaycomp:              delay compensation [in sec]
        """
        # init logger handle
        self.__log = logging.getLogger("local_trajectory_logger")

        # get valid export length of local trajectory
        self.__len_export = nmbr_export_points

        # --- Create SymQP-instance
        params_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '../../params/')
        input_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  '../../inputs/veh_dyn_info/')
        logging_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '../../logs/vp_sqp')

        self.__vp_sqp = vo.src.VelQP.VelQP(m=self.__len_export,
                                           sid='PerfSQP',
                                           params_path=params_path,
                                           input_path=input_path,
                                           logging_path=logging_path)

        self.__step_disc_len = stepsize_approx
        self.__s_glob_old = None

        # Ini of velocity guess
        self.__x0_ = 20 * np.ones((self.__vp_sqp.m,))
        self.__x0_guess = None

        # SQP-terminal constraint
        veh_turn = veh_turn

        # Specify entries in local gg to overwrite with conservative assumption and values
        self.__tire_end_idx = int(np.ceil(delaycomp * 50 / self.__step_disc_len))

        if self.__vp_sqp.sqp_stgs['b_var_friction']:
            self.__tire_end_mps2 = 3
        else:
            self.__tire_end_mps2 = self.__vp_sqp.sym_sc_['aymax_mps2_']

        self.__v_end_consv = np.sqrt(self.__tire_end_mps2 / (1 / veh_turn))

        self.__s_glob_last = glob_rl[-1][0]

        self.__Pmax = None

        self.__b_first_lap_done = False

        self.__ini_sqp = vo.src.IniSQPStatemachine.IniSQP()

        self.__vel_max = 0.0

        if self.__vp_sqp is not None and not self.__vp_sqp.sqp_stgs['b_var_friction']:
            self.__loc_axmax_mps2 = None
            self.__loc_aymax_mps2 = None

    def __del__(self) -> None:
        pass

    def update_dyn_parameters(self,
                              vel_max: float) -> None:
        """
        Update dynamic vehicle parameters (updated once per iteration).

        :param vel_max:         maximum velocity allowed by the behavior planner

        """

        self.__vel_max = vel_max

    def calc_vel_profile_follow(self,
                                action_id: str,
                                s_glob: float,
                                kappa: np.ndarray,
                                el_lengths: np.ndarray,
                                loc_gg: np.array,
                                v_obj: float,
                                vel_plan: float,
                                acc_plan: float,
                                safety_d: float,
                                obj_dist: float,
                                veh_length: float,
                                v_max_offset: float) -> tuple:
        """
        Calculates the velocity profile for a given trajectory, an target index (within the trajectory) as well as the
        velocity of the ego and the target vehicle.

        :param action_id:          name of the current action primitive
        :param s_glob:             global s-coordinate of the ego-vehicle
        :param kappa:              course of curvature in [1/m]
        :param el_lengths:         spatial distance between the curvature values [in m]
        :param loc_gg:             local gg diagram (lateral and longitudinal allowed acceleration per curvature value)
        :param v_obj:              velocity of target object [in mps]
        :param vel_plan:           velocity of the ego vehicle according to the last plan
        :param acc_plan:           acceleration of the ego vehicle according to the last plan
        :param safety_d:           safety distance to be maintained [in m]
        :param obj_dist:           distance to target vehicle [in m]
        :param veh_length:         length of vehicle [in m]
        :param v_max_offset:       allowed deviation of the planned velocity profile, to the current velocity
        :returns:
            * **vx** -             calculated velocity profile (same amount of data points as 'kappa' holds)
            * **too_close** -      flag holding 'True' if vehicle is within the safety distance to the other vehicle
            * **vel_bound** -      flag holding 'True' if velocity profile is not feasible for the given constraints

        """

        f_ini_kn_ = acc_plan * self.__vp_sqp.sym_sc_['m_t_'] + self.__vp_sqp.sym_sc_['c_res_'] * vel_plan ** 2 * 0.001

        # --- velocity profile for following
        # number of indices with max. velocity
        idx_vmax = int(np.ceil(
            (obj_dist - safety_d - veh_length) / self.__step_disc_len))
        if idx_vmax > self.__vp_sqp.m:
            idx_vmax = self.__vp_sqp.m
        elif idx_vmax < 0:
            idx_vmax = 0

        # --- Push as much as possible till safety distance
        # initialization of v_max
        vmax_mps = [self.__vel_max] * self.__vp_sqp.m
        # adaption of v_max to opponent
        if idx_vmax < self.__vp_sqp.m:
            # --- Fill with obj_velocity
            vmax_mps[idx_vmax:self.__vp_sqp.m] = [v_obj] * (self.__vp_sqp.m - idx_vmax)

        # Opponent braking assumption
        v_op = v_obj * np.ones((self.__vp_sqp.m,))
        for idx_op in range(1, self.__vp_sqp.m):
            rt = np.square(v_op[idx_op - 1]) - self.__vp_sqp.sym_sc_['axmax_mps2_'] * 2 * self.__step_disc_len
            if rt >= 0:
                v_op[idx_op] = np.sqrt(rt)
            else:
                v_op[idx_op] = 2.0  # small end velocity
                idx_op += 1
                break

        # Fill v_max-constraint with assumption that opponent brakes
        if idx_vmax + idx_op > self.__vp_sqp.m:
            idx_op = self.__vp_sqp.m - idx_vmax
        vmax_mps[idx_vmax:idx_vmax + idx_op] = v_op[0:idx_op]

        # --- Write also on hard constraint
        vmax_constraint = np.array(vmax_mps)

        # print('following ...')
        len_act_set = len(kappa)
        if len_act_set >= self.__vp_sqp.m:
            # if kappa-profile was too long, shorten
            kappa_ = kappa[0:self.__vp_sqp.m]
            delta_s_ = el_lengths[0:self.__vp_sqp.m - 1]

            if self.__vp_sqp.sqp_stgs['b_var_friction']:
                self.__loc_axmax_mps2 = loc_gg[:, 0][0:self.__vp_sqp.m]
                self.__loc_aymax_mps2 = loc_gg[:, 1][0:self.__vp_sqp.m]

        else:
            # if kappa-profile was too short, enlarge artificially
            kappa_ = kappa[0:len_act_set]
            # enlarge by last given kappa-value
            kappa_ = np.append(kappa_, [kappa_[-1]] * (self.__vp_sqp.m - len_act_set))
            delta_s_ = el_lengths[0:len_act_set - 1]
            # fill up delta_s artificially with last entry
            delta_s_ = np.append(delta_s_, [delta_s_[-1]] * (self.__vp_sqp.m - len_act_set))

            if self.__vp_sqp.sqp_stgs['b_var_friction']:
                self.__loc_axmax_mps2 = loc_gg[:, 0][0:len_act_set]
                self.__loc_axmax_mps2 = np.append(self.__loc_axmax_mps2,
                                                  [self.__loc_axmax_mps2[-1]]
                                                  * (self.__vp_sqp.m - len_act_set))
                self.__loc_aymax_mps2 = loc_gg[:, 1][0:len_act_set]
                self.__loc_aymax_mps2 = np.append(self.__loc_aymax_mps2,
                                                  [self.__loc_aymax_mps2[-1]]
                                                  * (self.__vp_sqp.m - len_act_set))

        # --- Velocity initialization
        self.__x0_guess = \
            self.__ini_sqp.get_v0(plan='f',
                                  action_id=action_id,
                                  m=self.__vp_sqp.m,
                                  b_print_sm=self.__vp_sqp.sqp_stgs['b_print_sm'])

        # --- Put most conservative assumption of minimum available friction into local gg
        if self.__loc_axmax_mps2 is not None:
            self.__loc_axmax_mps2[- self.__tire_end_idx:] = self.__tire_end_mps2
            self.__loc_aymax_mps2[- self.__tire_end_idx:] = self.__tire_end_mps2

        vx, _, qp_status = vo.src.online_qp. \
            online_qp(velqp=self.__vp_sqp,
                      v_ini=vel_plan,  # initial velocity constraint
                      kappa=kappa_,
                      delta_s=delta_s_,
                      ax_max=self.__loc_axmax_mps2,
                      ay_max=self.__loc_aymax_mps2,
                      x0_v=self.__x0_guess,  # initial velocity guess
                      v_max=vmax_mps,        # maximal velocity
                      v_end=self.__v_end_consv,
                      F_ini=f_ini_kn_,
                      s_glob=s_glob,
                      v_max_cstr=vmax_constraint)

        # --- Check if straight/follow line was infeasible --> trigger
        # infeasibility detection in ltpl
        if qp_status == -3:
            print('Velocity SQP triggered infeasibility detection in follow!')
            vx = np.zeros((self.__vp_sqp.m,))
        else:
            # --- Store solution
            self.__ini_sqp.set_vx(plan='f', action_id=action_id, vx=vx)

        too_close = False
        vel_bound = True
        if not abs(vx[0] - vel_plan) < v_max_offset:
            self.__log.warning("Velocity profile generation did not match the inserted boundary "
                               "conditions! (Action Set: " + action_id + ", Offset: %.2fm/s)" % (vx[0] - vel_plan))
            vel_bound = False

        # if kappa-profile was too long, append v=0 to match the expected v-profile
        if vx.shape[0] < len(kappa):
            vx = np.append(vx, [0] * (len(kappa) - self.__vp_sqp.m))

        return vx, too_close, vel_bound

    def calc_vel_profile(self,
                         action_id: str,
                         s_glob: float,
                         s: np.ndarray,
                         kappa: np.ndarray,
                         el_lengths: np.ndarray,
                         loc_gg: np.ndarray,
                         vel_plan: float,
                         acc_plan: float,) -> np.ndarray:
        """
        Calculates a velocity profile using the tire and motor limits.

        :param action_id:          name of the current action primitive
        :param s_glob:             global s-coordinate of the ego-vehicle
        :param s:                  s coordinates along the kappa profile
        :param kappa:              course of curvature in [1/m]
        :param el_lengths:         spatial distance between the curvature values [in m]
        :param loc_gg:             local gg diagram (lateral and longitudinal allowed acceleration per curvature value)
        :param vel_plan:           velocity of the ego vehicle according to the last plan
        :param acc_plan:           acceleration of the ego vehicle according to the last plan
        :returns:
            * **vx** -             calculated velocity profile (same amount of data points as 'kappa' holds)

        """

        vmax_mps = [self.__vel_max] * self.__vp_sqp.m

        # get Fini from current acceleration matching point
        f_ini_kn_ = acc_plan * self.__vp_sqp.sym_sc_['m_t_'] + self.__vp_sqp.sym_sc_['c_res_'] * vel_plan ** 2 * 0.001

        # take old solution as initial values for v-planner
        # only extract valid number of points that fit QP-dimensions
        # MPC-like: double last entries from previous solution for initial x0-guess
        # calculate travelled distance
        if self.__s_glob_old is None:
            self.__s_glob_old = s_glob
            push_idx = int(1)
        else:
            if np.round(s_glob) >= np.round(self.__s_glob_old):  # normal driving
                if np.round(s_glob) == np.round(self.__s_glob_old):
                    push_idx = 0
                else:
                    push_idx = int(
                        np.ceil((s_glob - self.__s_glob_old) / self.__step_disc_len))
            # new lap
            elif self.__s_glob_old > s_glob and s_glob - self.__s_glob_old < - 100:
                push_idx = int(np.ceil((s_glob + self.__s_glob_last - self.__s_glob_old)
                                       / self.__step_disc_len))
                self.__b_first_lap_done = True
            else:  # calculation of s_glob has twist
                push_idx = int(1)

        # --- Velocity initialization
        self.__x0_guess = \
            self.__ini_sqp.get_v0(plan='slr',
                                  action_id=action_id,
                                  m=self.__vp_sqp.m,
                                  b_print_sm=self.__vp_sqp.sqp_stgs['b_print_sm'])

        # --- Shift velocity guess
        self.__x0_[0:self.__vp_sqp.m - push_idx] = \
            self.__x0_guess[push_idx:self.__vp_sqp.m]
        # have last entry from x0_old several times in initial guess x0 (fill up)
        self.__x0_[self.__vp_sqp.m - push_idx:self.__vp_sqp.m] = \
            [self.__x0_guess[self.__vp_sqp.m - 1]] * push_idx

        # override s_glob_old
        self.__s_glob_old = s_glob

        len_act_set = len(kappa)
        if len_act_set >= self.__vp_sqp.m:
            # if kappa-profile was too long, shorten
            kappa_ = kappa[0:self.__vp_sqp.m]
            delta_s_ = el_lengths[0:self.__vp_sqp.m - 1]
            if self.__vp_sqp.sqp_stgs['b_var_friction']:
                self.__loc_axmax_mps2 = loc_gg[:, 0][0:self.__vp_sqp.m]
                self.__loc_aymax_mps2 = loc_gg[:, 1][0:self.__vp_sqp.m]
        else:
            # if kappa-profile was too short, enlarge artificially
            kappa_ = kappa[0:len_act_set]
            # enlarge by last given kappa-value
            kappa_ = np.append(kappa_,
                               [kappa_[-1]] * (self.__vp_sqp.m - len_act_set))

            delta_s_ = el_lengths[0:len_act_set - 1]
            # fill up delta_s artificially with last entry
            # if int(self.__vp_sqp.m - len_act_set - 1) is not 0:
            delta_s_ = np.append(delta_s_,
                                 [delta_s_[-1]] * (self.__vp_sqp.m - len_act_set))

            if self.__vp_sqp.sqp_stgs['b_var_friction']:
                self.__loc_axmax_mps2 = loc_gg[:, 0][0:len_act_set]
                self.__loc_axmax_mps2 = np.append(self.__loc_axmax_mps2,
                                                  [self.__loc_axmax_mps2[-1]]
                                                  * (self.__vp_sqp.m - len_act_set))
                self.__loc_aymax_mps2 = loc_gg[:, 1][0:len_act_set]
                self.__loc_aymax_mps2 = np.append(self.__loc_aymax_mps2,
                                                  [self.__loc_aymax_mps2[-1]]
                                                  * (self.__vp_sqp.m - len_act_set))

        # --- Put most conservative assumption of minimum available friction into local gg
        if self.__loc_axmax_mps2 is not None:
            self.__loc_axmax_mps2[- self.__tire_end_idx:] = self.__tire_end_mps2
            self.__loc_aymax_mps2[- self.__tire_end_idx:] = self.__tire_end_mps2

        if self.__vp_sqp.sqp_stgs['b_var_power']:
            if self.__b_first_lap_done:

                # --- Check if new lap arises in planning horizon and
                # --- begin/overwrite interpolation from s-coordinate = 0 m
                s_glob_interp = s_glob + np.cumsum(delta_s_)
                s_glob_interp[s_glob_interp > self.__vp_sqp.vpl.s_max_var_pwr_m] = \
                    s_glob_interp[s_glob_interp > self.__vp_sqp.vpl.s_max_var_pwr_m] - \
                    self.__vp_sqp.vpl.s_max_var_pwr_m

                self.__Pmax = self.__vp_sqp.vpl.f_pwr_intp(s_glob_interp)
            # --- Fade in power constraint on way to flying lap
            else:
                self.__Pmax = self.__vp_sqp.sym_sc_['Pmax_kW_'] * np.ones((self.__vp_sqp.m - 1,))
                # Heading towards new lap?
                if - self.__len_export * self.__step_disc_len < \
                        s_glob - self.__vp_sqp.vpl.s_max_var_pwr_m:
                    d_new_lap = self.__len_export - int(np.ceil((self.__vp_sqp.vpl.s_max_var_pwr_m - s_glob)
                                                                / self.__step_disc_len))
                    if d_new_lap < 1:
                        d_new_lap = 1
                    elif d_new_lap > self.__len_export - 1:
                        d_new_lap = self.__len_export - 1

                    # Fill points over finish line with power constraint in flying lap
                    self.__Pmax[- d_new_lap:] = \
                        self.__vp_sqp.vpl.f_pwr_intp(np.cumsum(delta_s_[- d_new_lap:]))

        # --- Call SQP solver
        vx, _, qp_status = vo.src.online_qp. \
            online_qp(velqp=self.__vp_sqp,
                      v_ini=vel_plan,  # initial velocity constraint
                      kappa=kappa_,
                      delta_s=delta_s_,
                      P_max=self.__Pmax,  # variable power constraint
                      ax_max=self.__loc_axmax_mps2,
                      ay_max=self.__loc_aymax_mps2,
                      x0_v=self.__x0_,  # initial velocity guess
                      v_max=vmax_mps,  # maximal velocity
                      v_end=self.__v_end_consv,
                      # initial applied powertrain constraint
                      F_ini=f_ini_kn_,
                      s_glob=s_glob)

        # --- Check if straight/follow line was infeasible --> trigger
        # infeasibility detection in ltpl
        if qp_status == -3 and \
                (action_id == 'straight' or action_id == 'follow'):
            print('Velocity SQP triggered infeasibility detection!')
            vx = np.zeros((self.__vp_sqp.m,))

        # --- Check if overtaking line was solved inaccurately or no solution present and
        # rm this solution by setting vx = 0
        if (qp_status == 2 or qp_status == -2) and \
                (action_id == 'left' or action_id == 'right'):
            vx = np.zeros((self.__vp_sqp.m,))
            print('Removed overtaking velocity profile due to inaccuracy!')
        elif qp_status == -3 and \
                (action_id == 'left' or action_id == 'right'):
            vx = np.zeros((self.__vp_sqp.m,))
            print('Removed overtaking velocity profile due to infeasibility!')

        # --- Store solution only if problem was not infeasible
        if qp_status != -3:
            self.__ini_sqp.set_vx(plan='slr', action_id=action_id, vx=vx)

        # if kappa-profile was too long, append v=0 to match the expected v-profile
        if vx.shape[0] < s.shape[0]:
            vx = np.append(vx, [0] * (s.shape[0] - self.__vp_sqp.m))
        # if kappa-profile was too short, cut too many entries in optimized v-profile
        elif vx.shape[0] > s.shape[0]:
            vx = vx[0:len_act_set]

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

        len_act_set = len(kappa)

        vmax_mps = 1 * np.ones((self.__vp_sqp.m,))

        if len_act_set >= self.__vp_sqp.m:
            # if kappa-profile was too long, shorten
            kappa_ = kappa[0:self.__vp_sqp.m]
            delta_s_ = el_lengths[0:self.__vp_sqp.m - 1]
            if self.__vp_sqp.sqp_stgs['b_var_friction']:
                self.__loc_axmax_mps2 = loc_gg[:, 0][0:self.__vp_sqp.m]
                self.__loc_aymax_mps2 = loc_gg[:, 1][0:self.__vp_sqp.m]
        else:
            # if kappa-profile was too short, enlarge artificially
            kappa_ = kappa[0:len_act_set]
            # enlarge by last given kappa-value
            kappa_ = np.append(kappa_, [kappa_[-1]] * (self.__vp_sqp.m - len_act_set))

            delta_s_ = el_lengths[0:len_act_set - 1]
            # fill up delta_s artificially with last entry
            # if int(self.__vp_sqp.m - len_act_set - 1) is not 0:
            delta_s_ = np.append(delta_s_, [delta_s_[-1]] * (self.__vp_sqp.m - len_act_set))

            if self.__vp_sqp.sqp_stgs['b_var_friction']:
                self.__loc_axmax_mps2 = loc_gg[:, 0][0:len_act_set]
                self.__loc_axmax_mps2 = np.append(self.__loc_axmax_mps2,
                                                  [self.__loc_axmax_mps2[-1]]
                                                  * (self.__vp_sqp.m - len_act_set))
                self.__loc_aymax_mps2 = loc_gg[:, 1][0:len_act_set]
                self.__loc_aymax_mps2 = np.append(self.__loc_aymax_mps2,
                                                  [self.__loc_aymax_mps2[-1]]
                                                  * (self.__vp_sqp.m - len_act_set))

        # Assume linear velocity decrease as initial guess
        self.__x0_ = np.array([v_start + x * (vmax_mps[- 1] - v_start)
                               / self.__vp_sqp.m for x in range(self.__vp_sqp.m)])
        # Value for problem infeasibility
        s_glob = -1.0

        vx = vo.src.online_qp. \
            online_qp(velqp=self.__vp_sqp,
                      v_ini=v_start,  # initial velocity constraint
                      kappa=kappa_,
                      delta_s=delta_s_,
                      P_max=self.__Pmax,  # variable power constraint
                      ax_max=self.__loc_axmax_mps2,
                      ay_max=self.__loc_aymax_mps2,
                      x0_v=self.__x0_,  # initial velocity guess
                      v_max=vmax_mps,  # maximal velocity
                      v_end=self.__v_end_consv,
                      F_ini=0,
                      s_glob=s_glob)[0]

        # if kappa-profile was too long, append v=0 to match the expected v-profile
        if vx.shape[0] < len_act_set:
            vx = np.append(vx, [0] * (len_act_set - self.__vp_sqp.m))
        # if kappa-profile was too short, cut too many entries in optimized v-profile
        elif vx.shape[0] > len_act_set:
            vx = vx[0:len_act_set]

        return vx
