import numpy as np


def closest_path_index(path: np.ndarray,
                       pos: tuple,
                       n_closest: int = 1) -> tuple:
    """
    Return the indices of the n closest coordinates to "pos" in the "path" array.

    :param path:            numpy array with columns holding x- and y-coordinates
    :param pos:             reference position
    :param n_closest:       (optional) number of closest indexes to be returned
    :returns:
        * **idx_array** -   list of indexes of points closest to reference-pos (sorted by index numbers, NOT distance
          to pos!)
        * **distances2** -  squared distances to all path coordinates

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        23.01.2019

    """

    # calculate squared distances between path array and reference poses
    distances2 = np.power(path[:, 0] - pos[0], 2) + np.power(path[:, 1] - pos[1], 2)

    # get indexes of smallest values in sorted order
    idx_array = sorted(np.argpartition(distances2, n_closest)[:n_closest])

    return idx_array, distances2


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
