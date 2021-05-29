#!/usr/bin/env python3
import zmq
import time
import os
import sys
import numpy as np
import signal
import json
import configparser

# custom packages
import trajectory_planning_helpers as tph

# custom modules
mod_local_traj_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(mod_local_traj_path)
import graph_ltpl

"""
Script allows the generation of a simple object list. Two options are available:

* Init the contained class and retrieve an object list via call of the function 'get_objectlist()'
* Call the script, this initialized the contained class and publishes the object list via the parameter. ZMQ interface

:Authors:
    * Tim Stahl <tim.stahl@tum.de>

:Created on:
    13.11.2018
"""


def create_obj_list_sender():
    context = zmq.Context()
    sock = context.socket(zmq.PUB)
    sock.bind("tcp://*:47209")
    return sock, context


def signal_handler(sig, frame):
    print("Clearing all zones and objects...")
    socket.send_string("v2x_to_all", zmq.SNDMORE)
    socket.send_json([])
    time.sleep(0.5)
    socket.send_string("v2x_to_all", zmq.SNDMORE)
    socket.send_json([])
    time.sleep(0.5)
    print("Closing ZMQ socket...")
    socket.close()
    context.term()
    time.sleep(0.5)
    print("Shutdown complete!")
    sys.exit(0)


# listen to "Control + C"
signal.signal(signal.SIGINT, signal_handler)


class ObjectlistDummy(object):
    """
    Simple class, that loads a race line and allows to retrieve the position of the vehicle at arbitrary time stamps
    (the passed time is tracked by the class).

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        13.11.2018

    """

    def __init__(self,
                 dynamic: bool,
                 vel_scale: float = 0.5,
                 s0: float = 0.0):
        """

        :param dynamic:     boolean, if 'True', a vehicle driving on a loaded race line is simulated, if 'False', the
                            (hardcoded) static positions are returned
        :param vel_scale:   velocity factor for the dynamic replay of the loaded race line (1.0 being loaded speed)
        :param s0:          s-coordinate the dynamic vehicle should be initialized to

        """

        self.__dynamic = dynamic

        if dynamic:
            # read race line
            toppath = os.path.dirname(os.path.realpath(__file__))
            sys.path.append(toppath)

            # top level path (module directory)
            toppath = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            sys.path.append(toppath)

            track_param = configparser.ConfigParser()
            if not track_param.read(toppath + "/../../params/driving_task.ini"):
                raise ValueError('Specified online parameter config file does not exist or is empty!')

            track_specifier = json.loads(track_param.get('DRIVING_TASK', 'track'))

            globtraj_param_path = toppath + "/../../inputs/traj_ltpl_cl/traj_ltpl_cl_" + track_specifier + ".csv"

            # load data from csv files
            refline, t_width_right, t_width_left, normvec_normalized, alpha, length_rl, vel_rl, kappa_rl \
                = graph_ltpl.imp_global_traj.src.import_globtraj_csv.\
                import_globtraj_csv(import_path=globtraj_param_path)

            # get race line
            self.__raceline = refline + normvec_normalized * alpha[:, np.newaxis]
            self.__s_rl = np.cumsum(length_rl)

            # calculate race line heading
            self.__psi_rl = tph.calc_head_curv_num.calc_head_curv_num(path=self.__raceline,
                                                                      el_lengths=length_rl,
                                                                      is_closed=True)[0]
            self.__psi_rl = np.where(self.__psi_rl < 0.0, self.__psi_rl + np.pi * 2, self.__psi_rl)

            self.__pos = self.__raceline[0, :]
            self.__vel_rl = vel_rl * vel_scale
        else:
            self.__s_rl = None
            self.__vel_rl = None
            self.__psi_rl = None
            self.__raceline = None

        self.__pos_index = 0
        self.__tic = time.time()
        self.s = s0

    def get_objectlist(self):
        """
        Lightweight function allowing to integrate forward on a given path with velocity profile.

        :returns:
            * **obj_list** -     dummy object-list

        :Authors:
            * Tim Stahl <tim.stahl@tum.de>

        :Created on:
            13.11.2018
        """

        if self.__dynamic:
            # measure iteration duration
            toc = time.time() - self.__tic
            self.__tic = time.time()

            t = 0.0
            dt = 0.001

            while t < toc:
                self.s += np.interp(self.s, self.__s_rl, self.__vel_rl) * dt
                t += dt

                if self.s >= self.__s_rl[-1]:
                    self.s = 0.0

            # calculate new point and velocity between the two determined points
            pos_out = [np.interp(self.s, self.__s_rl, self.__raceline[:, 0]),
                       np.interp(self.s, self.__s_rl, self.__raceline[:, 1])]

            psi_out = np.interp(self.s, self.__s_rl, self.__psi_rl)
            if psi_out > np.pi:
                psi_out -= 2 * np.pi

            vel_est = np.interp(self.s, self.__s_rl, self.__vel_rl)

            obj_list = [{'X': pos_out[0], 'Y': pos_out[1], 'theta': psi_out, 'type': 'physical',
                         'id': 1, 'length': 5.0, 'v': vel_est}]
        else:
            # define dummy objects for testing at specified position
            objA = {'X': 127, 'Y': 82, 'theta': 0.0, 'type': 'physical',
                    'id': 1, 'length': 5.0, 'width': 2.5, 'v': 0.0}
            # objB = {'X': 11.1, 'Y': 53, 'theta': 0.0, 'type': 'car', 'form': 'rectangle',
            #         'id': 1, 'length': 5.0, 'width': 2.5, 'v_x': 0.0}
            # objC = {'X': 2.9, 'Y': 41.7, 'theta': 0.0, 'type': 'car', 'form': 'rectangle',
            #         'id': 1, 'length': 5.0, 'width': 2.5, 'v_x': 0.0}
            # objD = {'X': -4.0, 'Y': 32.0, 'theta': 0.0, 'type': 'car', 'form': 'rectangle',
            #         'id': 1, 'length': 5.0, 'width': 2.5, 'v_x': 0.0}
            # objE = {'X': -11.7, 'Y': 19.3, 'theta': 0.0, 'type': 'car', 'form': 'rectangle',
            #         'id': 1, 'length': 5.0, 'width': 2.5, 'v_x': 0.0}

            # fuse objects in list in order to obtain proper format
            obj_list = [objA]  # , objB, objC, objD, objE]

        return obj_list


if __name__ == "__main__":
    # create socket
    socket, context = create_obj_list_sender()

    # init object list dummy class
    obj_dummy = ObjectlistDummy(dynamic=True,
                                vel_scale=0.5)

    while True:
        # get objectlist
        obj_dummy_list = obj_dummy.get_objectlist()

        # send object list via ZMQ
        socket.send_string("v2x_to_all", zmq.SNDMORE)
        socket.send_json(obj_dummy_list)
        print("sending object list... (s: " + str(int(obj_dummy.s)) + "m, time: " + str(time.time()) + ")")

        # wait some time to limit send frequency
        time.sleep(0.1)
