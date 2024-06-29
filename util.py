import pickle
import numpy as np
from typing import Union
from pathlib import Path

def save_pickle(data: object, filename: Union[str, Path]):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=-1)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results


def splitx(string: str) -> tuple[int, ...]:
    """
    Receive a string like "5x6x7" (no spaces) and return a tuple of ints, in
    this case, (5, 6, 7).
    :param string: A string of numbers separated with "x".
    :return: Return a list of int
    """
    return tuple(map(int, string.split('x')))


def rot_matrix(yaw_pitch_roll: Union[np.ndarray, list]) -> np.ndarray:
    """
    Create rotation matrix using Tait–Bryan angles in Z-Y-X order.
    See Wikipedia. Use:
        O exito x aponta para a direita
        O eixo y aponta para baixo
        O eixo z aponta para a frente


    Examples
    --------
    >> x, y, z = point
    >> mat = rot_matrix(yaw, pitch, roll)
    >> mat @ (x, y, z)

    :param yaw_pitch_roll: the rotation (yaw, pitch, roll) in rad.
    :return: A 3x3 matrix of rotation for (z,y,x) vector
    """
    cos_rot = np.cos(yaw_pitch_roll)
    sin_rot = np.sin(yaw_pitch_roll)

    # pitch
    mat_x = np.array(
        [[1, 0, 0],
         [0, cos_rot[1], -sin_rot[1]],
         [0, sin_rot[1], cos_rot[1]]]
    )
    # yaw
    mat_y = np.array(
        [[cos_rot[0], 0, sin_rot[0]],
         [0, 1, 0],
         [-sin_rot[0], 0, cos_rot[0]]]
    )
    # roll
    mat_z = np.array(
        [[cos_rot[2], -sin_rot[2], 0],
         [sin_rot[2], cos_rot[2], 0],
         [0, 0, 1]]
    )

    return mat_y @ mat_x @ mat_z


def xyz2hcs(x, y, z) -> tuple[float, float]:
    """
    Convert from cartesian system to horizontal coordinate system in radians
    :param float x: Coordinate from X axis
    :param float y: Coordinate from Y axis
    :param float z: Coordinate from Z axis
    :return: (azimuth, elevation) - in rad
    """
    # z-> x,
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = np.arctan2(x, z)
    elevation = np.arcsin(-y / r)
    return azimuth, elevation



def get_borders(*, coord_nm: Union[tuple, np.ndarray] = None, shape=None, thickness=1):
    """
    frame must be shape==(C, N, M)
    :param coord_nm:
    :param shape:
    :param thickness:
    :return: shape==(C, thickness*(2N+2M))
    """
    if coord_nm is None:
        assert shape is not None
        coord_nm = np.mgrid[range(shape[0]), range(shape[1])]
        C = 2
    else:
        C = coord_nm.shape[0]

    left = coord_nm[:, :, 0:thickness].reshape((C, -1))
    right = coord_nm[:, :, :- 1 - thickness:-1].reshape((C, -1))
    top = coord_nm[:, 0:thickness, :].reshape((C, -1))
    bottom = coord_nm[:, :- 1 - thickness:-1, :].reshape((C, -1))

    return np.c_[top, right, bottom, left]
