
import sys
sys.path.insert(0, '..')

import jefpy as jp
import numpy as np


class PointSourceTest(jp.PointSource):

    def E_retarded_coordinates(self, R, tau):
        return R * tau + jp.norm_keepdim(R)

    def B_retarded_coordinates(self, R, tau):
        return R * tau + jp.norm_keepdim(R)


def test_Pointretarded_coordinates():
    source = PointSourceTest()
    R, tau = source.retarded_coordinates([10, 1, 1.0], 0.0)
    assert R.shape == (3,)
    assert tau.shape == (1,)

    R, tau = source.retarded_coordinates(np.ones((23, 1, 3, 4, 3)), 0.0)
    assert R.shape == (23, 1, 3, 4, 3)
    assert tau.shape == (23, 1, 3, 4, 1)


def test_broadcasting():
    source = PointSourceTest()

    assert source.E([1.0, 0, 0], 1.0).shape == (3,)
    assert source.E([[1.0, 0, 0], [0, 0, 0]], 1.0).shape == (2, 3)
    assert source.B([[1.0, 0, 0], [0, 0, 0]], [1, 2, 3, 4]).shape == (4, 2, 3)
    assert source.B([1.0, 0, 0], [1, 2, 3, 4]).shape == (4, 3)

    source_1 = PointSourceTest()
    source_2 = PointSourceTest()

    sources = jp.SourceCollection(sources=(source_1, source_2))

    assert sources.E([1.0, 0, 0], 1.0).shape == (3,)
    assert sources.E([[1.0, 0, 0], [0, 0, 0]], 1.0).shape == (2, 3)
    assert sources.B([[1.0, 0, 0], [0, 0, 0]], [1, 2, 3, 4]).shape == (4, 2, 3)
    assert sources.B([1.0, 0, 0], [1, 2, 3, 4]).shape == (4, 3)



def test_SourceCollection():
    t1 = PointSourceTest(location=(1.0, 2, 3), cut_off=2.0)
    t2 = PointSourceTest(location=(1.0, 2, 3), cut_off=3.0)
    jp.SourceCollection((t1, t2))
    jp.SourceCollection(t1)
    source = jp.SourceCollection((t1, t2))
   # assert source._sum(np.ones((2, 4, 1, 3)), 0.0, 'E').shape == (2, 4, 1, 3)
    #TODO way more is needed here.


def test_SourceCollection2():
    t1 = PointSourceTest(location=(1.0, 2, 3), cut_off=2.0)
    t2 = PointSourceTest(location=(1.0, 2, 3), cut_off=3.0)
    t3 = PointSourceTest(location=(1.0, 2, 3), cut_off=4.0)
    sources = t1 + t2
    sources_2 = sources + t3
    sources_3 = sources + sources + sources_2


def test_ElectricDipole_static():

    p = np.array([0, 0, 1.0])
    dipole = jp.ElectricDipole(p=p, location=[0, 0, 0])

    r = np.array([1, 10.0, 1])
    r_unit = r / np.linalg.norm(r)
    E = 1 / (8.854187e-12 * 4 * np.pi) * (3 * np.dot(p, r_unit) * r_unit - p) \
            / np.linalg.norm(r) ** 3

    E2 = jp.Constants.K_e * (3 * np.dot(p, r_unit) * r_unit - p) / \
         np.linalg.norm(r) ** 3


    assert np.all(np.isclose(dipole.E(r, 0.987), E))
    assert np.all(dipole.B(r, 123.123) == np.zeros(3))


def test_ElectricDipole_oscillating():
    dipole_settings = {
        'power': 1.0,
        'orientation': [0, 0.2, 1],
        'freq': 200e6,
        'location': [0.0, 0, 0]}

    dipole = jp.ElectricDipole.oscillator(**dipole_settings)

    for t in np.linspace(0, 20e-9, 13):
        orientation = dipole_settings['orientation'] / np.linalg.norm(dipole_settings['orientation'])
        E_unit = abs(dipole.E((1.0, 0, 0), t) / np.linalg.norm(dipole.E((1.0, 0, 0), t)))
        # The E field is parallel to p, for r perpendicular to p at any t.
        #print(dipole.E((1.0, 0, 0), t))
        assert all(np.isclose(orientation, E_unit))

def test_MagneticDipole_static():

    m = np.array([0, 0, 2.0])
    dipole = jp.MagneticDipole(m=m, location=[0, 0, 0])

    r = np.array([1, 10, 1])
    r_unit = r / np.linalg.norm(r)
    B = 1.256637062e-6 * (3 * (np.dot(m, r_unit)) * r_unit - m) / \
        (4 * np.pi) / np.linalg.norm(r) ** 3

    assert np.all(np.isclose(dipole.B(r, 0.123), B))
    assert np.all(dipole.E(r, 123.33) == np.zeros(3))

def test_MagneticDipole_oscillating():
    pass


def test_Observer_single():
    dipole_settings = {
        'p': [0, 0, 1.0],
        'location': [0, 0, 0]}
    dipole = jp.ElectricDipole(**dipole_settings)
    observer = jp.Observer([1e30, 1e30, 1e30], dipole)
    assert observer.E(0.0).shape == (3,)
    assert observer.B(0.0).shape == (3,)
    assert np.linalg.norm(observer.E(0.0)) < 1e-6
    assert np.linalg.norm(observer.B(0.0)) < 1e-6
    assert np.linalg.norm(observer.B(134.234)) == np.linalg.norm(observer.B(0.0))


def test_Observer_map():
    dipole_settings = {
        'p': [0, 0, 1.0],
        'location': [0, 0, 0]}
    dipole = jp.ElectricDipole(**dipole_settings)

    x = np.linspace(-1.0, 3.0, 4)
    y = np.linspace(-1.0, 2.0, 5)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Z = np.ones_like(X) * 1000
    R = np.stack((X, Y, Z), axis=-1)

    observer = jp.Observer(R, dipole)
    assert observer.E(0.0).shape == (4, 5, 3)
    assert observer.B(0.0).shape == (4, 5, 3)

    observer = jp.Observer(R, dipole, observable='norm')
    assert observer.E(0.0).shape == (4, 5)
    assert observer.B(0.0).shape == (4, 5)


def test_Observer_list():
    dipole_settings = {
        'p': [0, 0, 1.0],
        'location': [0, 0, 0]}
    dipole = jp.ElectricDipole(**dipole_settings)

    r1 = [1e30, 1e30, 1e30]
    r2 = [-2.0, -2.0, -2.0]
    R = (r1, r2)

    observer = jp.Observer(R, dipole)
    assert observer.E(0.0).shape == (2, 3)
    assert observer.B(0.0).shape == (2, 3)

    observer = jp.Observer(R, dipole, observable='norm')
    assert observer.E(0.0).shape == (2,)
    assert observer.B(0.0).shape == (2,)

def test_Observer_broadcasting():
    dipole_settings = {
        'p': [0, 0, 1.0],
        'location': [0, 0, 0]}
    dipole = jp.ElectricDipole(**dipole_settings)

    observer = jp.Observer([10, 1, 1.0], dipole, observable=(0, 1, 2))
    observer2 = jp.Observer([10, 1, 1.0], dipole, observable=(0, 1, 2))
    assert np.all(observer.E(0.0) == observer2.E(0.0))

    observer = jp.Observer([10, 1, 1.0], dipole, observable=(0, 2))
    assert observer.E(0.0).size == 2

    observer = jp.Observer([10, 1, 1.0], dipole, observable=1)
    assert observer.E(0.0).size == 1

    observer = jp.Observer([10, 1, 1.0], dipole, observable='norm')
    assert observer.E(0.0).size == 1

    observer = jp.Observer([10, 1, 1.0], dipole)
    assert observer.E([0, 1, 3, 6.0]).shape == (4, 3)

    observer = jp.Observer([[10, 1, 1.0], [10, 1, 1.0]], dipole)
    assert observer.E([0, 1, 3, 5]).shape == (4, 2, 3)

    observer.r(0.0)
    observer.r([0.0, 1.0])


def test_Observer_fancy():
    def r_magnet(t): return np.array([5, 5, 5]) * t
    def r_observer(t): return np.array([1, 1, 1]) * t
    magnet = jp.MagneticDipole(location=r_magnet, m=(1, 2, 3))
    observer = jp.Observer(r_observer, magnet)
    assert np.all(observer.B(1.2) > observer.B(4.2))


def test_HarmonicOscillator():
    osc = jp.HarmonicOscillator(amplitude=5.0, freq=2.0, phase=0.0)
    assert np.isclose(osc.f(0.0), osc.f(1 / 2.00))
    osc_2pi = jp.HarmonicOscillator(amplitude=5.0, freq=2.0, phase=2 * np.pi)
    assert np.isclose(osc.f(0.1), osc_2pi.f(0.1))
    assert np.isclose(osc.df_dt(0.1), osc_2pi.df_dt(0.1))
    assert np.isclose(osc.d2f_dt2(0.1), osc_2pi.d2f_dt2(0.1))


def test_WireSegment():
    line = ((0, 0, 0), (1, 1, 0))
    wiresegment = jp.WireSegment(line, 5.0)
    wiresegment.E((0, 10, 20), 123.02)
    wiresegment.B((0, 10, 20), 123.02)


def test_WireSegment_callbacks():
    def line(t): return np.array((0, 1, 2)) * t, np.array((1, 3, 2)) * t
    osc = jp.HarmonicOscillator()
    wiresegment = jp.WireSegment(line, osc.f, osc.df_dt)
    wiresegment.E((0, 10, 20), 123.02)
    wiresegment.B((0, 10, 20), 123.02)


def test_Wire():
    wiresegment1 = jp.WireSegment(((0, 0, 0), (1, 1, 0)), 5.0)
    wiresegment2 = jp.WireSegment(((0, 0, 0), (1, 0, 1)), 3.0)
    wiresegment3 = jp.WireSegment(((0, 0, 0), (1, 1, 1)), 2.0)
    wire = jp.Wire((wiresegment1, wiresegment2, wiresegment3))
    wire.E((0, 10, 20), 123.02)
    wire.B((0, 10, 20), 123.02)


def test_Wire_from_curve():
    def curve(s, t):
        return np.array((12, 2, 2)) * s ** 2  + t
    wire = jp.Wire.curve(curve, I=5.0)
    wire.E((0, 10, 20), 123.02)
    wire.B((0, 10, 20), 123.02)


def test_Wire_polygon():
    polygon= np.array(((0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)))
    wire = jp.Wire.polygon(polygon, I=5.0)
    wire.E((0, 10, 20), 123.02)
    wire.B((0, 10, 20), 123.02)


def test_Wire_circle():
    settings = {
        'center': [0, 0, 0.0],
        'radius': 1.5,
        'num_segments': 20}
    wire = jp.Wire.circle(**settings)
    wire.E([0, 1, 2], 0)
    wire.B([0, 1, 2], 0)


def test_Uniform():

    def E(t):
        return np.array([1, 2, 3.0]) * t

    def B(t):
        return np.array([1, 2, 3.0]) * np.cos(t)

    source = jp.Uniform(E=E, B=B)
    assert np.all(source.E([1.0, 0, 0], 1.0) == source.E([2.0, 0, 0], 1.0))

    assert source.E([1.0, 0, 0], 1.0).shape == (3,)
    assert source.E([[1.0, 0, 0], [0, 0, 0]], 1.0).shape == (2, 3)
    assert source.B([[1.0, 0, 0], [0, 0, 0]], [1, 2, 3, 4]).shape == (4, 2, 3)
    assert source.B([1.0, 0, 0], [1, 2, 3, 4]).shape == (4, 3)

    source = jp.Uniform(E=[1, 2, 3], B=[1, 5, 3])
    assert np.all(source.E([1.0, 0, 0], 1.0) == source.E([2.0, 0, 0], 1.0))

    assert source.E([1.0, 0, 0], 1.0).shape == (3,)
    assert source.E([[1.0, 0, 0], [0, 0, 0]], 1.0).shape == (2, 3)
    assert source.B([[1.0, 0, 0], [0, 0, 0]], [1, 2, 3, 4]).shape == (4, 2, 3)
    assert source.B([1.0, 0, 0], [1, 2, 3, 4]).shape == (4, 3)

def test_static_magnetic_dipole():

    dipole = jp.MagneticDipole(location=[0,0,0], m=[1, 1, 0])
    assert np.all(np.isclose(dipole.B([0, 1.0, 0], 0.0),
                             1.256637062121e-6 / (4 * np.pi) * np.array((-1, 2, 0))))
