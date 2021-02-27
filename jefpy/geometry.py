"""
Convenience module for sampling purposes. Transformations to curvilinear
coordinates are also available.
"""

from jefpy.utils import default, is_iter
from jefpy.math import dot, norm
import numpy as np


class Surface:
    """
    TODO refactor using inheritance
    Class for creating meshed surfaces. The class methods set self.XYZ.
    """

    def __init__(self, u=None, v=None):
        """
        Object initialization only makes sense from a class method,
        which determines the type of surface.
        u: (N) array, spans the first dimension of the surface.
        v: (M) array, spans the second dimension of the surface.
        U: (N x M) array, meshgrid.
        V: (N x M) array, meshgrid.
        XYZ: (N x M x 3) array, cartesian coordinates of the surface.
        """
        self.u = default(u, np.linspace(0.0, 1.0, 10))
        self.v = default(v, np.linspace(0.0, 1.0, 10))
        self.U, self.V = np.meshgrid(self.u, self.v, indexing='ij')
        self.UV = np.stack((self.U, self.V), axis=-1)
        self.XYZ = np.empty((len(self.u), len(self.v), 3))

    @classmethod
    def cartesian(cls, x=None, y=None, z=None):
        """
        Creates a surface that is in the xy, yz, or xz plane. What plane
        is created is determined by what input argument is a float and which
        ones are iterable. Two have to be iterable and one has to be a float
        for a valid surface to be created.
        """
        x = default(x, np.linspace(0.0, 1.0, 10))
        y = default(y, np.linspace(0.0, 1.0, 10))
        z = default(z, 0.0)
        # Some tricks to determine if uv = xy, yz, or xz

        axes = np.array((is_iter(x), is_iter(y), is_iter(z)))
        u, v = np.array((x, y, z), dtype=object)[axes]
        w = np.array((x, y, z), dtype=object)[~axes]
        self = cls(u=u, v=v)
        self.XYZ[:, :, axes] = np.stack((self.U, self.V), axis=-1)
        self.XYZ[:, :, ~axes] = (w * np.ones_like(self.U))[:, :, np.newaxis]
        return self

    @classmethod
    def cylinder(cls, radius=1.0, phi=None, z=None, axis='z'):
        """
        Creates a surface that is the outside of a cylinder. By default the
        cylinder points in the z-direction, but it can be oriented along the
        x or the y axis.

        :param radius: float, single value setting the cylinder radius
        :param phi: 1D-array, span of angles
        :param z: 1D-array, span of the 'z' axis.
        :param axis: str, sets the orientation ('z'-axis) of the cylinder
        """
        phi = default(phi, np.linspace(0.0, 2*np.pi, 10))
        z = default(z, np.linspace(0.0, 1.0, 10))
        surf = cls(u=phi, v=z)
        i, j, k = {'z': (0, 1, 2), 'y': (2, 0, 1), 'x': (1, 2, 0)}[axis]
        surf.XYZ[..., i] = radius * np.cos(surf.U)
        surf.XYZ[..., j] = radius * np.sin(surf.U)
        surf.XYZ[..., k] = surf.V
        return surf


    @classmethod
    def sphere(cls, radius=1.0, theta=None, phi=None, center=(0, 0, 0)):
        """
        Creates a spherical surface located anywhere in space.

        :param radius: flaot, single vale setting the sphere radius.
        :param theta: 1D-array, angle w.r.t the z-axis.
        :param phi: 1D-array, angle in the xy plane.
        :param center: (3) array, Location in space of the center of the sphere
        """
        phi = default(phi, np.linspace(0.0, 2*np.pi, 10))
        theta = default(theta, np.linspace(0.0, np.pi, 10))
        self = cls(u=theta, v=phi)
        self.XYZ[..., 0] = radius * np.sin(self.U) * np.cos(self.V)
        self.XYZ[..., 1] = radius * np.sin(self.U) * np.sin(self.V)
        self.XYZ[..., 2] = radius * np.cos(self.U)
        self.XYZ += np.array(center)
        return self

    def flat_coords(self):
        X, Y, Z = self.split_mesh()
        return X.flatten(), Y.flatten(), Z.flatten()

    def split_mesh(self):
        X, Y, Z = np.split(self.XYZ, 3, axis=2)
        return X, Y, Z


class Volume:
    """
    Class for creating meshed volumes. The class methods set self.XYZ.
    """

    def __init__(self, u=None, v=None, w=None):
        """
        Object initialization only makes sense from a class method,
        which determines the type of surface.

        :param u: (N) array, spans the first dimension of the volume.
        :param v: (M) array, spans the first dimension of the volume.
        :param w: (L) array, spans the first dimension of the volume.
        """
        self.u = default(u, np.linspace(0.0, 1.0, 10))
        self.v = default(v, np.linspace(0.0, 1.0, 10))
        self.w = default(w, np.linspace(0.0, 1.0, 10))
        self.X, self.Y, self.Z = \
            np.meshgrid(self.u, self.v, self.w)
        self.XYZ = np.empty((len(self.u), len(self.v), 3))

    @classmethod
    def cartesian(cls, x=None, y=None, z=None):
        """
        Creates a cartesian meshed rectangle.
        :param x: 1D-array, span of x.
        :param y: 1D-array, span of y.
        :param z: 1D-array, span of z.
        """
        self = cls(u=x, v=y, w=z)
        self.XYZ = np.stack((self.X, self.Y, self.Z), axis=-1)
        return self


def to_spherical(r, F):
    """
    Transforms a vector field from cartesian to spherical coordinates. r and
    F can have any shape, as long as the last dimension represents one
    particular 3D vector.
    :param r: (..., 3) array, cartesian positions.
    :param F: (..., 3) array, cartesian vector field.
    :return: (..., 3) array, vector field in spherical coordinates.
    """
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    l = norm(r)
    l_xy = np.sqrt(x ** 2 + y ** 2)

    # unit vectors
    rho = r / l[..., np.newaxis]
    theta_x = z * x / l / l_xy
    theta_y = z * y / l / l_xy
    theta_z = - l_xy / l
    theta = np.stack((theta_x, theta_y, theta_z), axis=-1)
    phi = np.stack((-y / l, x / l, x * 0), axis=-1)

    return np.stack((dot(F, rho), dot(F, theta), dot(F, phi)), axis=-1)



def _to_cylindrical(r, F, axis='z'):
    """
    Transforms a vector field from cartesian to cylindrical coordinates. r and
    F can have any shape, as long as the last dimension represents one
    particular 3D vector.
    :param r: (..., 3) array, cartesian positions.
    :param F: (..., 3) array, cartesian vector field.
    :param axis: str, pecify the orientation of the cylinder.
    :return: (..., 3) array, vector field in cylindrical coordinates.
    """
    i, j, k = {'z': (0, 1, 2), 'y': (2, 0, 1), 'x': (1, 2, 0)}[axis]
    l = np.sqrt(r[..., i] ** 2 + r[..., j] ** 2)

    # unit vectors
    zero = np.zeros_like(l)
    rho = np.stack((r[..., i] / l, r[..., j] / l, zero), axis=-1)
    rho = rho[..., (i, j, k)]
    theta = np.stack((-r[..., j] / l, r[..., i] / l, zero), axis=-1)
    theta = theta[..., (i, j, k)]

    return np.stack((dot(F, rho), dot(F, theta), F[..., k]), axis=-1)


def to_cylindrical_x(r, F):
    """
    Transforms a vector field from cartesian to cylindrical coordinates,
    with the cylinder oriented around the x-axis. r and  F can have any shape,
    as long as the last dimension represents one particular 3D vector.
    :param r: (..., 3) array, cartesian positions.
    :param F: (..., 3) array, cartesian vector field.
    :return: (..., 3) array, vector field in cylindrical coordinates.
    """
    return _to_cylindrical(r, F, 'x')


def to_cylindrical_y(r, F):
    """
    Transforms a vector field from cartesian to cylindrical coordinates,
    with the cylinder oriented around the y-axis. r and  F can have any shape,
    as long as the last dimension represents one particular 3D vector.
    :param r: (..., 3) array, cartesian positions.
    :param F: (..., 3) array, cartesian vector field.
    :return: (..., 3) array, vector field in cylindrical coordinates.
    """
    return _to_cylindrical(r, F, 'y')


def to_cylindrical_z(r, F):
    """
    Transforms a vector field from cartesian to cylindrical coordinates,
    with the cylinder oriented around the z-axis. r and  F can have any shape,
    as long as the last dimension represents one particular 3D vector.
    :param r: (..., 3) array, cartesian positions.
    :param F: (..., 3) array, cartesian vector field.
    :return: (..., 3) array, vector field in cylindrical coordinates.
    """
    return _to_cylindrical(r, F, 'z')


def to_cylindrical(r, F):
    """
    Transforms a vector field from cartesian to cylindrical coordinates, 
    with the cylinder oriented around the z-axis. r and F can have any shape, 
    as long as the last dimension represents one particular 3D vector.
    :param r: (..., 3) array, cartesian positions.
    :param F: (..., 3) array, cartesian vector field.
    :return: (..., 3) array, vector field in cylindrical coordinates.
    """
    return _to_cylindrical(r, F, 'z')
