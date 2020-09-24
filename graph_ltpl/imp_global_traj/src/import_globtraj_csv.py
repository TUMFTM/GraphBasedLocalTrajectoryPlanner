import numpy as np


def import_globtraj_csv(import_path: str) -> tuple:
    """
    Read in csv-flavoured file holding map and global race line data with the following columns:
    (x_ref_m;y_ref_m;width_center_m;x_normvec;y_normvec;alpha;s_raceline;vel_rl)

    :param import_path:             path pointing to the file to be imported
    :returns:
        * **refline** -             x and y coordinate of reference line
        * **width right/left** -    width to track bounds at given ref-line coordinates in meters
        * **normvec_normalized** -  x and y components of normized normal vector at given ref-line coordinates
        * **alpha** -               alpha parameter (offset in meters to refline along normal vector) defining race line
        * **length_raceline** -     length of race-line spline segments (from current race-line point to adjacent one)

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        24.09.2018

    """

    # ------------------------------------------------------------------------------------------------------------------
    # IMPORT DATA ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # load data from csv file (closed; assumed order listed below)
    # x_ref_m, y_ref_m, width_right_m, width_left_m, x_normvec_m, y_normvec_m, alpha_m, s_racetraj_m,
    # psi_racetraj_rad, kappa_racetraj_radpm, vx_racetraj_mps, ax_racetraj_mps2
    csv_data_temp = np.loadtxt(import_path, delimiter=';')

    # get refline
    refline = csv_data_temp[:-1, 0:2]

    # get widths right/left
    width_right = csv_data_temp[:-1, 2]
    width_left = csv_data_temp[:-1, 3]

    # get normized normal vectors
    normvec_normalized = csv_data_temp[:-1, 4:6]

    # get raceline alpha
    alpha = csv_data_temp[:-1, 6]

    # get racline segment lengths
    length_rl = np.diff(csv_data_temp[:, 7])

    # get kappa at raceline points
    kappa_rl = csv_data_temp[:-1, 9]

    # get velocity at raceline points
    vel_rl = csv_data_temp[:-1, 10]

    return refline, width_right, width_left, normvec_normalized, alpha, length_rl, vel_rl, kappa_rl


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
