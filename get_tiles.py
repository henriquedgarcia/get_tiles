import argparse
from abc import ABC, abstractmethod
from math import pi
from typing import Union

import numpy as np

np.set_printoptions(precision=3)

from matplotlib import pyplot as plt

from util import splitx, rot_matrix, get_borders


class Viewport:
    base_normals: np.ndarray
    fov: tuple
    px_in_vp: np.ndarray
    yaw_pitch_roll: np.ndarray = np.array([0, 0, 0])
    _mat_rot: np.ndarray
    _rotated_normals: np.ndarray
    _is_in_vp: bool

    def __init__(self, yaw_pitch_roll: tuple, fov: tuple):
        """
        Viewport Class used to extract view pixels in projections.
        The vp is an image as numpy array with shape (H, M, 3).
        That can be RGB (matplotlib, pillow, etc) or BGR (opencv).

        :param frame yaw_pitch_roll: (600, 800) for 800x600px
        :param fov: (fov_v, fov_h) in rad. Ex: "np.array((pi/2, pi/2))" for (90°x90°)
        """
        self.fov = fov
        self.make_base_normals()
        self.yaw_pitch_roll = np.asarray(yaw_pitch_roll)

    def is_viewport(self, x_y_z: np.ndarray) -> bool:
        """
        Check if the plane equation is true to viewport
        x1 * m + y1 * n + z1 * z < 0
        If True, the "point" is on the viewport
        Obs: is_in só retorna true se todas as expressões forem verdadeiras

        :param x_y_z: A 3D Point list in the space [(x, y, z), ...].T, shape == (3, ...)
        :return: A boolean         belong = np.all(inner_product <= 0, axis=0).reshape(self.shape)

        """
        inner_prod = self.rotated_normals.T @ x_y_z
        self.px_in_vp = np.all(inner_prod <= 0, axis=0)
        self._is_in_vp = np.any(self.px_in_vp)
        return self._is_in_vp

    def make_base_normals(self) -> None:
        """
        Com eixo entrando no observador, rotação horário é negativo e anti-horária
        é positivo. Todos os ângulos são radianos.

        O exito x aponta para a direita
        O eixo y aponta para baixo
        O eixo z aponta para a frente

        Deslocamento para a direita e para cima e horário é positivo.

        O viewport é a região da esfera que faz intersecção com 4 planos que passam pelo
          centro (4 grandes círculos): cima, direita, baixo e esquerda.
        Os planos são definidos tal que suas normais (N) parte do centro e apontam na mesma direção a
          região do viewport. Ex: O plano de cima aponta para cima, etc.
        Todos os píxeis que estiverem abaixo do plano {N(x,y,z) dot P(x,y,z) <= 0}
        O plano de cima possui incinação de FOV_Y / 2.
          Sua normal é x=0,y=sin(FOV_Y/2 + pi/2), z=cos(FOV_Y/2 + pi/2)
        O plano de baixo possui incinação de -FOV_Y / 2.
          Sua normal é x=0,y=sin(-FOV_Y/2 - pi/2), z=cos(-FOV_Y/2 - pi/2)
        O plano da direita possui incinação de FOV_X / 2. (para direita)
          Sua normal é x=sin(FOV_X/2 + pi/2),y=0, z=cos(FOV_X/2 + pi/2)
        O plano da esquerda possui incinação de -FOV_X/2. (para direita)
          Sua normal é x=sin(-FOV_X/2 - pi/2),y=0, z=cos(-FOV_X/2 - pi/2)

        :return:
        """
        fov_y_2, fov_x_2 = np.array(self.fov) / (2, 2)
        pi_2 = np.pi / 2

        self.base_normals = np.array([[0, -np.sin(fov_y_2 + pi_2), np.cos(fov_y_2 + pi_2)],  # top
                                      [0, -np.sin(-fov_y_2 - pi_2), np.cos(-fov_y_2 - pi_2)],  # bottom
                                      [np.sin(fov_x_2 + pi_2), 0, np.cos(fov_x_2 + pi_2)],  # left
                                      [np.sin(-fov_x_2 - pi_2), 0, np.cos(-fov_x_2 - pi_2)]]).T  # right

    @property
    def rotated_normals(self) -> np.ndarray:
        self._rotated_normals = rot_matrix(self.yaw_pitch_roll) @ self.base_normals
        return self._rotated_normals


class Projection(ABC):
    viewport: Viewport
    yaw_pitch_roll: np.ndarray
    projection: np.ndarray  # A RGB image
    vptiles: dict

    def __init__(self, tiling: str, proj_res: str, fov: str):
        # Create the viewport
        self.fov = np.deg2rad(splitx(fov)[::-1])

        # Create the projection
        self.shape = np.array(splitx(proj_res)[::-1], dtype=int)
        self.proj_coord_nm = np.mgrid[range(self.shape[0]), range(self.shape[1])]

        # Create the tiling
        self.tiling = np.array(splitx(tiling)[::-1], dtype=int)
        self.n_tiles = self.tiling[0] * self.tiling[1]
        self.tile_shape = (self.shape / self.tiling).astype(int)
        self.tile_position_list = np.array([(n, m)
                                            for n in range(0, self.shape[0], self.tile_shape[0])
                                            for m in range(0, self.shape[1], self.tile_shape[1])])

        # Get tiles borders
        self.tile_border_base = get_borders(shape=self.tile_shape)

    @property
    @abstractmethod
    def nfaces(self):
        pass

    def get_tile_borders_nm(self, idx: int):
        # projection agnostic
        return self.tile_border_base + self.tile_position_list[idx].reshape(2, -1)

    def get_tile_array(self, tile, array):
        n, m = self.tile_position_list[tile]
        h, w = self.tile_shape
        if array.shape[0] >= 3:
            return array[..., n:n + h, m:m + w]
        else:
            return array[n:n + h, m:m + w]

    @staticmethod
    @abstractmethod
    def nm2xyz(nm_coord: np.ndarray, shape: np.ndarray, face: int):
        pass

    @staticmethod
    @abstractmethod
    def xyz2nm(nm_coord: np.ndarray, shape: np.ndarray):
        pass

    @abstractmethod
    def get_vptiles(self, yaw_pitch_roll) -> list[int]:
        pass

    @abstractmethod
    def get_projection(self):
        pass


class ERP(Projection):
    @property
    def nfaces(self):
        return 1

    def __init__(self, tiling: str, proj_res: str, fov: str):
        super().__init__(tiling, proj_res, fov)
        self.proj_coord_xyz = self.nm2xyz(self.proj_coord_nm, self.shape)

        self.tiles_borders_xyz = []
        for tile in range(self.n_tiles):
            borders_nm = self.get_tile_borders_nm(tile)
            self.tiles_borders_xyz.append(self.nm2xyz(nm_coord=borders_nm, shape=self.shape))

    def get_vptiles(self, yaw_pitch_roll) -> dict[int, str]:
        """

        :param yaw_pitch_roll: The coordinate of center of VP.
        :return:
        """
        if tuple(self.tiling) == (1, 1):
            return {0: '100.00%'}

        self.viewport = Viewport(yaw_pitch_roll, self.fov)
        self.vptiles = {}
        n_pix = self.tile_shape[0] * self.tile_shape[1]

        for tile in range(self.n_tiles):
            tile_coord_xyz = self.get_tile_array(tile, self.proj_coord_xyz)
            if self.viewport.is_viewport(tile_coord_xyz.reshape([3, -1])):
                count_pix = np.sum(self.viewport.px_in_vp)
                self.vptiles[tile] = f'{100 * count_pix / n_pix:.2f}%'
                # plt.imshow(self.viewport.px_in_vp.reshape(self.tile_shape));plt.show()

        return self.vptiles

    def get_projection(self):
        self.projection = np.zeros(self.shape, dtype='uint8')

        for tile in range(self.n_tiles):
            n, m = self.get_tile_borders_nm(tile)
            if tile in self.vptiles:
                self.projection[n, m] = 255
            else:
                self.projection[n, m] = 100

        self.viewport.is_viewport(self.proj_coord_xyz.reshape((3, -1)))
        belong = self.viewport.px_in_vp.reshape(self.proj_coord_xyz.shape[1:])
        self.projection[belong] = 200

        return self.projection

    @staticmethod
    def nm2xyz(nm_coord: np.ndarray, shape: np.ndarray, face: int = 0):
        """
        ERP specific.

        :param face: 0 for ERP
        :param nm_coord: shape==(2,...)
        :param shape: (N, M)
        :return:
        """
        azimuth = ((nm_coord[1] + 0.5) / shape[1] - 0.5) * 2 * np.pi
        elevation = ((nm_coord[0] + 0.5) / shape[0] - 0.5) * -np.pi

        z = np.cos(elevation) * np.cos(azimuth)
        y = -np.sin(elevation)
        x = np.cos(elevation) * np.sin(azimuth)

        xyz_coord = np.array([x, y, z])
        return xyz_coord

    @staticmethod
    def xyz2nm(xyz_coord: np.ndarray, shape: np.ndarray = None, round_nm: bool = False):
        """
        ERP specific.

        :param xyz_coord: [[[x, y, z], ..., M], ..., N] (shape == (N,M,3))
        :param shape: the shape of projection that cover all sphere
        :param round_nm: round the coords? is not needed.
        :return:
        """
        if shape is None:
            shape = xyz_coord.shape[:2]

        N, M = shape[:2]

        r = np.sqrt(np.sum(xyz_coord ** 2, axis=0))

        elevation = np.arcsin(xyz_coord[1] / r)
        azimuth = np.arctan2(xyz_coord[0], xyz_coord[2])

        v = elevation / pi + 0.5
        u = azimuth / (2 * pi) + 0.5

        n = v * N - 0.5
        m = u * M - 0.5

        if round_nm:
            n = np.mod(np.round(n), N)
            m = np.mod(np.round(m), M)

        return np.array([n, m])


class CMP(Projection):
    def __init__(self, tiling: str, proj_res: str, fov: str):
        super().__init__(tiling, proj_res, fov)

        # Create faces structures
        face_h, face_w = face_shape = (self.shape / (2, 3)).astype(int)
        face_array_mn = np.mgrid[range(face_h), range(face_w)]

        self.face_position_list = np.array([(n, m)
                                            for n in range(0, self.shape[0], face_h)
                                            for m in range(0, self.shape[1], face_w)])

        self.proj_coord_xyz = self.nm2xyz(self.proj_coord_nm)

        self.tiles_borders_xyz = []
        for tile in range(self.n_tiles):
            borders_nm = self.get_tile_borders_nm(tile)
            borders_xyz = self.proj_coord_xyz[:, borders_nm[0, :], borders_nm[1, :]]  # testar
            self.tiles_borders_xyz.append(borders_xyz)

    @property
    def nfaces(self):
        return 6

    @staticmethod
    def nm2xyz(nm_coord: np.ndarray, proj_shape: np.ndarray = None, face: int = None):
        # nm_coord is only for face
        u = v = np.array([])
        z, y, x = 0, 0, 0

        def face2vu():
            nonlocal u, v
            face_nm_array = np.mgrid[range(face_h), range(face_w)]
            u = 2 * (face_nm_array[1] + 0.5) / face_w - 1  # (-1, 1)
            v = -2 * (face_nm_array[0] + 0.5 - face_h) / face_h - 1  # (-1, 1)

        def vu2xyz():
            nonlocal z, y, x
            nonlocal u, v
            if face == 0:
                # left no rotate
                x = -np.ones(u.shape)
                y = -v
                z = u
            elif face == 1:
                # center no rotate
                x = u
                y = -v
                z = np.ones(u.shape)
            elif face == 2:
                # right no rotate
                x = np.ones(u.shape)
                y = -v
                z = -u
            elif face == 3:
                # down rotate  90° anti-clockwise
                w = np.zeros(u.shape)
                a = np.array([u, v, w])
                u1, v1, w3 = (a.transpose([1, 2, 0]) @ rot_matrix([0, 0, pi / 2])).transpose([2, 0, 1])

                x = u1
                y = np.ones(u.shape)
                z = v1
            elif face == 4:
                # back rotate 90° clockwise
                w = np.zeros(u.shape)
                a = np.array([u, v, w])
                u1, v1, w3 = (a.transpose([1, 2, 0]) @ rot_matrix([0, 0, -pi / 2])).transpose([2, 0, 1])

                x = -u1
                y = -v1
                z = -np.ones(u.shape)
            elif face == 5:
                # up rotate 90° anti-clockwise
                w = np.zeros(u.shape)
                a = np.array([u, v, w])
                u1, v1, w3 = (a.transpose([1, 2, 0]) @ rot_matrix([0, 0, pi / 2])).transpose([2, 0, 1])

                x = u1
                y = -np.ones(u.shape)
                z = -v1

        if face is None:
            if proj_shape is None:
                proj_shape = np.array(nm_coord.shape[-2:])

            face_h, face_w = (proj_shape / (2, 3)).astype(int)
            face2vu()

            face_position_list = np.array([(n, m)
                                           for n in range(0, proj_shape[0], face_h)
                                           for m in range(0, proj_shape[1], face_w)])
            proj_coord_xyz = np.zeros((3, proj_shape[0], proj_shape[1]))
            for face in range(6):
                vu2xyz()
                n, m = face_position_list[face]
                proj_coord_xyz[:, n:n + face_h, m:m + face_w] = np.array([x, y, z])

            return proj_coord_xyz

        else:
            assert isinstance(face, int)
            assert isinstance(proj_shape, np.ndarray)
            face_h, face_w = proj_shape / (2, 3)
            face2vu()
            vu2xyz()
            return np.array([x, y, z])

    @staticmethod
    def xyz2nm(xyz_coord: np.ndarray, shape: np.ndarray = None, round_nm: bool = False, face: int = 0):
        x, y, z = xyz_coord

        f = 0
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        azimuth = np.arctan2(x, z)
        elevation = np.arcsin(-y / r)
        u = azimuth / (2 * np.pi) + 0.5
        v = -elevation / np.pi + 0.5
        f, m, n = 0, 0, 0
        return f, m, n

    get_vptiles = ERP.get_vptiles
    get_projection = ERP.get_projection

    def ea2nm(self, ea_coord: np.ndarray, shape: Union[tuple, np.ndarray] = None):
        """

        :param ea_coord: [elevation, azimuth].shape == (2, proj_h, proj_w)
        :param shape: Shape from the projection (proj_h, proj_w)
        :return:
        """
        shape = shape if shape is not None else ea_coord[0].shape
        face_shape = shape[0] / 2, shape[1] / 3

        xyz = ea2xyv(ea_coord)
        uv = self.xyv2uv(xyz)
        nm = uv2nm(ea_coord)

    @staticmethod
    def xyv2uv(xyv_coord: np.ndarray, faces):
        x, y, z = xyv_coord
        vu_coord = np.zeros((3,) + x.shape)
        abs_xyv_coord = np.abs(xyv_coord)
        vu_coord[0][faces == 0] = x/z
        vu_coord[1][faces == 0] = x/-y

        vu_coord[0][faces == 0] = -y
        vu_coord[1][faces == 0] = z
        vu_coord[0][faces == 1] = -y
        vu_coord[1][faces == 1] = x
        vu_coord[0][faces == 2] = -y
        vu_coord[1][faces == 2] = -z
        vu_coord[0][faces == 3] = x
        vu_coord[1][faces == 3] = -z
        vu_coord[0][faces == 4] = x
        vu_coord[1][faces == 4] = -y
        vu_coord[0][faces == 5] = x
        vu_coord[1][faces == 5] = z

    @staticmethod
    def get_faces(ea_coord):
        # find face
        # l f r  (0, 1, 2)
        # b r t  (3, 4, 5)
        shape = ea_coord[0].shape
        face_h, face_w = shape[0] / 2, shape[1] / 3

        faces = np.zeros(ea_coord[0].shape) + 4  # default is rear
        faces[ea_coord[0] > pi / 4] = 5  # Top
        faces[ea_coord[0] < -pi / 4] = 3  # Down
        faces[-pi / 4 <= ea_coord[0] <= pi / 4 and -3 * pi / 4 < ea_coord[1] <= -pi / 4] = 0  # left
        faces[-pi / 4 <= ea_coord[0] <= pi / 4 and -pi / 4 < ea_coord[1] <= pi / 4] = 1  # front
        faces[-pi / 4 <= ea_coord[0] <= pi / 4 and pi / 4 < ea_coord[1] <= 3 * pi / 4] = 2  # right
        return faces







def ea2xyv(hcs_coord: np.ndarray):
    elevation, azimuth = hcs_coord
    z = np.cos(elevation) * np.cos(azimuth)
    y = -np.sin(elevation)
    x = np.cos(elevation) * np.sin(azimuth)
    return np.asarray([x, y, z])

def xyv2ea(xyz_coord: np.ndarray) -> tuple[float, float]:
    x, y, z = xyz_coord
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = np.arctan2(x, z)
    elevation = np.arcsin(-y / r)
    return elevation, azimuth










def test():
    # erp '144x72', [array([0, 0]), array([144,  72]), array([288, 144]), array([432, 216]), array([576, 288]), array([720, 360]), array([864, 432]), array([1008,  504]), array([1152,  576]), array([1296,  648])]
    # cmp '108x72', [array([0, 0]), array([108,  72]), array([216, 144]), array([324, 216]), array([432, 288]), array([540, 360]), array([648, 432]), array([756, 504]), array([864, 576]), array([972, 648])]

    # proj = ERP('6x4', f'1152x768', '100x100')
    proj = CMP('6x4', f'972x648', '100x90')

    # for i in [30]:
    for frame, i in enumerate(range(0, -360, -10)):
        tiles = proj.get_vptiles(np.deg2rad((0, 0, i)))
        projection = proj.get_projection()
        plt.imsave(f'teste/teste_{frame}.png', projection)

        print(f'The viewport touch the tiles {tiles}.')


def main():
    if proj == 'erp':
        projection = ERP(tiling, f'1296x648', fov)
    elif proj == 'cmp':
        projection = CMP(tiling, f'972x648', fov)
    else:
        raise ValueError('The -proj must be "erp" or "cmp"')

    tiles = projection.get_vptiles(np.deg2rad(yaw_pitch_roll))

    if out is not None:
        projection = projection.get_projection()
        plt.imsave(f'{out}', projection)

    import json
    print(json.dumps(tiles, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description='Get the tiles seen in viewport.')
    parser.add_argument('-proj', default='erp', metavar='PROJECTION', help='The projection [erp|cmp]')
    parser.add_argument('-fov', default='100x90', metavar='FOV', help=f'The Field of View in degree. Ex: 100x90')
    parser.add_argument('-tiling', default='3x2', metavar='TILING', help=f'The tiling of projection. Ex: 3x2.')
    parser.add_argument('-coord', default=['0', '0', '0'], nargs=3, metavar=('YAW', 'PITCH', 'ROLL'), help=f'The center of viewport in degree ex: 15,25,-30')
    parser.add_argument('-out', default=None, metavar='OUTPUT_FILE', help=f'Save the projection marks to OUTPUT_FILE file.')

    args = parser.parse_args()
    proj = args.proj
    fov = args.fov
    tiling = args.tiling
    yaw_pitch_roll: list = list(map(float, args.coord))
    out = args.out

    main()
