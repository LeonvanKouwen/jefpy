import sys
sys.path.insert(0, '..')

import jefpy as jp
import numpy as np

def test_Surface_cartesian():
    setup = {
        'x': np.linspace(0.0, 1.0, 13),
        'y': np.linspace(0.0, 1.0, 10),
        'z': 5.0}
    surface = jp.Surface.cartesian(**setup)
    assert surface.XYZ.shape == (13, 10, 3)

    setup = {
        'x': np.linspace(0.0, 3.0, 4),
        'y': 5.0,
        'z': np.linspace(0.0, 2.0, 3)}
    surface = jp.Surface.cartesian(**setup)
    assert np.all(surface.XYZ[..., 1] == 5.0)


def cylinder(axis):
    setup = {
        'phi': np.linspace(0.0, np.pi, 11),
        'z': np.linspace(0.0, 1.0, 10),
        'radius': 13.9,
        'axis': axis}
    return jp.Surface.cylinder(**setup)


def test_Surface_cylinder():
    surf = cylinder('z')
    R = surf.XYZ[..., 0] ** 2 + surf.XYZ[..., 1] ** 2
    assert np.all(np.isclose(np.mean(R - np.mean(R)), 0.0))

    surf = cylinder('x')
    R = surf.XYZ[..., 1] ** 2 + surf.XYZ[..., 2] ** 2
    assert np.all(np.isclose(np.mean(R - np.mean(R)), 0.0))


def test_to_cylider():
    r = np.ones((10, 5, 5, 3))
    F = np.ones((10, 5, 5, 3)) + 2.0
    jp.to_cylindrical(r, F)
    assert np.all(jp.to_cylindrical(r, F) == jp.to_cylindrical_z(r, F))

def test_to_spherical_1D():
    x = np.linspace(1.0, 2.0, 6)
    y = np.linspace(1.0, 2.0, 6)
    z = np.linspace(0.0, 0.0, 6)
    r = np.stack((x, y, z), axis=-1)
    F = r
    F_spher = jp.to_spherical(r, F)
    Fr = np.linalg.norm(r, axis=-1)
    assert np.all(np.isclose(Fr, F_spher[..., 0]))
    assert np.all(np.isclose(0.0, F_spher[..., 1]))
    assert np.all(np.isclose(0.0, F_spher[..., 2]))


def test_to_spherical_2D():
    x = np.linspace(-1.0, 3.0, 4)
    y = np.linspace(-1.0, 2.0, 3)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.ones_like(X) * 1000
    r = np.stack((X, Y, Z), axis=-1)
    F = r
    F_spher = jp.to_spherical(r, F)
    Fr = np.linalg.norm(r, axis=-1)
    assert np.all(np.isclose(Fr, F_spher[..., 0]))
    assert np.all(np.isclose(0.0, F_spher[..., 1]))
    assert np.all(np.isclose(0.0, F_spher[..., 2]))

def test_to_spherical_3D():
    x = np.linspace(-1.0, 3.0, 4)
    y = np.linspace(-1.0, 2.0, 3)
    z = np.linspace(-1.0, 2.0, 3)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    r = np.stack((X, Y, Z), axis=-1)

    r_norm = np.linalg.norm(r, axis=-1)
    F = r / r_norm[..., np.newaxis]
    F_spher = jp.to_spherical(r, F)

    assert np.all(np.isclose(1.0, F_spher[..., 0]))
    assert np.all(np.isclose(0.0, F_spher[..., 1]))
    assert np.all(np.isclose(0.0, F_spher[..., 2]))

