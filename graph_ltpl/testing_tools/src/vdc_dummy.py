#!/usr/bin/env python3
import numpy as np


def vdc_dummy(pos_est: list,
              last_s_course: np.ndarray,
              last_path: np.ndarray,
              last_vel_course: np.ndarray,
              iter_time: float):
    """
    Lightweight function allowing to integrate forward on a given path with velocity profile.

    :param pos_est:         estimated position of the vehicle
    :param last_s_course:   s-course of the last planned trajectory
    :param last_path:       coordinates of the last planned trajectory
    :param last_vel_course: velocity-profile of the last planned trajectory
    :param iter_time:       time to be driven forward along given trajectory
    :returns:
        * **pos_out** -     new position
        * **vel_est** -     estimated velocity at new position

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        13.11.2018

    """

    if last_path.shape[0] > 2:
        # -- get s value from current pose on trajectory ---------------------------------------------------------------
        distances2 = (np.power(last_path[:, 0] - pos_est[0], 2)
                      + np.power(last_path[:, 1] - pos_est[1], 2))
        idx_nb = sorted(np.argpartition(distances2, 2)[:2])
        nb_1 = last_path[idx_nb[0], :]

        # -- distance from pos_est to next point on trajectory ---------------------------------------------------------
        dist_s = np.sqrt(np.power(nb_1[0] - pos_est[0], 2)
                         + np.power(nb_1[1] - pos_est[1], 2))

        s = dist_s + last_s_course[idx_nb[0]]
        t = 0
        dt = 0.001

        while t < iter_time:
            s += max(np.interp(s, last_s_course, last_vel_course) * dt, 0.0001)
            t += dt

        # -- calculate new point and velocity between the two determined points ----------------------------------------
        pos_out = [np.interp(s, last_s_course, last_path[:, 0]),
                   np.interp(s, last_s_course, last_path[:, 1])]

        vel_est = np.interp(s, last_s_course, last_vel_course)
    else:
        pos_out = pos_est
        vel_est = last_vel_course[0]

    return pos_out, vel_est
