"""
The heart of the jefpy package. All sources and their convenience containers are
defined here.
"""

from abc import ABC, abstractmethod
import numpy as np
import uuid
from collections import OrderedDict
from dataclasses import dataclass

from jefpy.math import HarmonicOscillator
from jefpy.math import cross, norm, dot_keepdim, norm_keepdim, norm_keepshape, vec_split
from jefpy.utils import broadcast_spacetime, default, match_shape, JProperty


@dataclass(init=False)
class Constants:
    C: float = 299792458
    EPS_0: float = 8.8541878128e-12
    MU_0: float = 1.25663706212e-6
    K_e: float = 1 / (4 * np.pi * EPS_0)
    K_u: float = MU_0 / (4 * np.pi)


class Source(ABC):
    """
    Template for any source. If a source is only uw or electric,
    an array of zeros should be returned for the non-existent field.
    Allowed types of r, t depends on the implementation. See main documentation.

    Note about the common arguments r, t:
    r: (..., 3): position(s) of observer anywhere in the shared spacetime frame.
    t: float or (..., 1): time of the observation in shared spacetime frame.
    """

    def E(self, r, t=.0):
        """ Returns the electric field at location r at time t. """
        raise NotImplementedError

    def B(self, r, t=.0):
        """ Returns the uw field at location r at time t. """
        raise NotImplementedError

    @broadcast_spacetime
    def S(self, r, t=.0):
        """ Returns the Poynting vector at location r at time t. """
        return 1.0 / Constants.MU_0 * cross(self.E(r, t), self.B(r, t))

    def __add__(self, addition):
        if isinstance(addition, Source):
            return SourceCollection((self, addition))
        elif isinstance(addition, SourceCollection):
            addition[str(uuid.uuid4())] = self
            return addition

class SourceCollection(OrderedDict):
    """
    Special dictionary of source objects. Besides regular dictionary
    operations, the methods E, B and S are available, which give
    the summed fields of all sources.

    Can be nested with SourceCollections. This is very useful when
    making groups of sources.

    A note on inheritance from dict, which is sometimes frowned upon.
    Because the standard mechanisms such as __get__ and __set__ are
    not mingled with here, so for this purpose it is alright imho.
    """

    def __init__(self, sources=None, observable='uvw', transform=None):
        """
        sources: a Source object or a collection of Source objects.
        observable: str / tuple / int
            Applies a tunable observation to the field.
            Options are 'uvw', 'norm', 'uv', 'uw', 'vw', or a tuple specifying
            which spatial dimension need to be taken, or an int representing one
            single dimension. Note that without transformation, uvw = xyz. However
            with transformation, uvw represent the new coordinate system dimensions.
        transform: callback
            Coordinate system transformation to apply to the vector field.
        """
        if sources is None:
            sources = {}
        elif isinstance(sources, (list, tuple, set)):
            sources = {uuid.uuid4(): source for source in sources}
        elif isinstance(sources, Source):
            sources = {uuid.uuid4(): sources}
        super().__init__(sources)

        self.observable = observable
        self.transform = transform or (lambda r, F: F)

    def _sum(self, r, t, field_type, transform=None, observable=None):
        """
        Returns the field of all sources combined (superposition) at location                      
        r at time t. 
        field_type: str, 'E', 'B', 'S'
        """
        F = np.zeros_like(r)
        for source in self.values():
            if isinstance(source, Source):
                F = F + getattr(source, field_type)(r, t)
            else:
                F = F + source._sum(r, t, field_type)

        # It is possible to overwrite the default output manipulations.
        transform = default(transform, self.transform)
        observable = default(observable, self.observable)

        F = transform(r, F)
        F = self._observe(F, observable=observable)
        return F

    @broadcast_spacetime
    def E(self, r, t=.0, transform=None, observable=None):
        """ Returns the electric field at location r at time t. """
        return self._sum(r, t, 'E', transform=transform, observable=observable)

    @broadcast_spacetime
    def B(self, r, t=.0, transform=None, observable=None):
        """ Returns the uw field at location r at time t. """
        return self._sum(r, t, 'B', transform=transform, observable=observable)

    @broadcast_spacetime
    def S(self, r, t=.0, transform=None, observable=None):
        """ Returns the Poynting vector field at location r at time t. """
        return self._sum(r, t, 'S', transform=transform, observable=observable)

    @staticmethod
    def _observe(F, observable='uvw'):
        if observable == 'uvw':
            return F
        elif observable == 'norm':
            return norm(F)
        elif observable == 'uv':
            return F[..., (0, 1)]
        elif observable == 'uw':
            return F[..., (0, 2)]
        elif observable == 'vw':
            return F[..., (1, 2)]
        elif type(observable) in (tuple, int):
            return F[..., observable]
        else:
            raise ValueError(observable, " is not a valid observable.")

    def get_source_attributes(self, attribute, as_array=True):
        #TODO, add denesting function?
        attributes = {}
        for key, source in self.items():
            if isinstance(source, Source):
                attributes[key] = getattr(source, attribute)
            else:
                attributes[key] = source.get_source_attributes(attribute, as_array=as_array)
        if as_array:
            try:
                # Assume that attr is a JProperty and thus callable.
                return np.array([attr() for attr in attributes.values()])
            except KeyError:
                return np.array(attributes.values())
        else:
            return attributes

    def get_locations_array(self):
        #BACKWARD COMPATABILITY, REMOVE AT SOME POINT
        return self.get_source_attributes("location")

    def __add__(self, addition):
        if isinstance(addition, Source):
            self[uuid.uuid4()] = addition
        elif isinstance(addition, SourceCollection):
            self.update(addition)
        return self

    def update(self, addition, **kwargs):
        if isinstance(addition, (list, tuple)):
            addition = {uuid.uuid4(): value for value in addition}
        super().update(addition)


class PointSource(Source):
    """
    Base class for most sources. Assumes the location is one point in space.
    The location can be set by an array or by a callable returning an array
    such that the source can move through space.

    Retarded coordinates are used. This means that for implementation
    of a source it be assumed to be in the origin. The retarded coordinates are:
    R: (..., 3): vector from source to observer.
    tau: float or (..., 1): retarded time at observer.
    """

    location = JProperty((0, 0, 0))
    cut_off = JProperty(1e-12)

    def __init__(self, location=None, cut_off=None):
        self.location = location
        self.cut_off = cut_off
        #super().__init__()

    @broadcast_spacetime
    def E(self, r, t=.0):
        """ Returns the electric field at location r at time t. """
        return self.E_retarded_coordinates(*self.retarded_coordinates(r, t))

    @broadcast_spacetime
    def B(self, r, t=.0):
        """ Returns the uw field at location r at time t. """
        return self.B_retarded_coordinates(*self.retarded_coordinates(r, t))

    @broadcast_spacetime
    def V(self, r, t=.0):
        """ Returns the electrostatic potential at location r at time t."""
        return self.V_retarded_coordinates(*self.retarded_coordinates(r, t))

    def E_retarded_coordinates(self, R, tau):
        """ Computes the electrostatic potential from retarded coordinates. """
        raise NotImplementedError

    def B_retarded_coordinates(self, R, tau):
        """ Computes the electrostatic potential from retarded coordinates. """
        raise NotImplementedError

    def V_retarded_coordinates(self, R, tau):
        """ Computes the electrostatic potential from retarded coordinates. """
        raise NotImplementedError

    def retarded_coordinates(self, r, t):
        """ Creates the retarded coordinates at the requested space-time locations. """
        R = np.array(r) - self.location(t)
        tau = t - norm_keepdim(R) / Constants.C
        # Mind the broadcasting of various dimensionalities.
        # Introduces nan's for retarded coordinates too close to the source.
        R[norm_keepshape(R) < self.cut_off(t)] = np.nan
        return R, tau


class ElectricDipole(PointSource):
    """ Infinitely small electric dipole."""

    p = JProperty(np.zeros(3))
    dp_dt = JProperty(np.zeros(3))
    d2p_dt2 = JProperty(np.zeros(3))

    def __init__(self, p=None, dp_dt=None, d2p_dt2=None, **kwargs):
        """
        :param location: callable that returns a 3-array, or a 3-array. Sets
                         the (dynamic) location of the source.
        :param p: callable, function of t returning the dipole moment.
        :param dp_dt: callable, function of returning the derivative of the
                     dipole moment.
        :param d2p_dt2: callable, function of t returning the second
                       derivative of the dipole moment.
        :param cut_off: float, exclusion distance around the source location.
        """
        super().__init__(**kwargs)
        self.p = p
        self.dp_dt = dp_dt
        self.d2p_dt2 = d2p_dt2

    @classmethod
    def oscillator(cls, power=1.0, freq=1e9, orientation=(0, 0, 1), phase=0.0, **kwargs):
        """
        Harmonically oscillating electric dipole.
        :param location: callable that returns a 3-array, or a 3-array. Sets
                         the (dynamic) location of the source.
        :param power: float, total irradiated power.
        :param freq: float, frequency
        :param orientation: 3-array, direction of the dipole moment.
                            Magnitude is ignored.
        :param phase: float
        :param cut_off: float, distance to source that returns nan.
        :return:
        """

        # TODO: Consider making these properties all quasi-static
        orientation = np.array(orientation) / norm(np.array(orientation))
        p0 = (np.array(orientation)
              * np.sqrt(power * 12 * np.pi * Constants.C / Constants.MU_0) /
              (freq * 2 * np.pi) ** 2)
        osc = HarmonicOscillator(amplitude=p0, freq=freq, phase=phase)
        return cls(p=osc.f, dp_dt=osc.df_dt, d2p_dt2=osc.d2f_dt2, **kwargs)

    def E_retarded_coordinates(self, R, tau):
        """ Computes the electric field from retarded coordinates. """
        R_norm, R_unit = vec_split(R)
        p = self.p(tau)
        dp_dt = self.dp_dt(tau)
        d2p_dt2 = self.d2p_dt2(tau)
        near1 = Constants.C * (3 * dot_keepdim(R_unit, p) * R_unit - p) / R_norm ** 3
        near2 = (3 * dot_keepdim(R_unit, dp_dt) * R_unit - dp_dt) / R_norm ** 2
        far = (cross(R_unit, cross(R_unit, d2p_dt2)) / (Constants.C * R_norm))
        return Constants.K_e / Constants.C * (near1 + near2 + far)

    def B_retarded_coordinates(self, R, tau):
        """ Computes the uw field from retarded coordinates. """
        R_norm, R_unit = vec_split(R)
        dp_dt = self.dp_dt(tau)
        d2p_dt2 = self.d2p_dt2(tau)
        near = cross(dp_dt, R_unit) / R_norm ** 2 / Constants.C
        far = cross(d2p_dt2, R_unit) / R_norm / Constants.C ** 2
        return Constants.K_u * (near + far)


class MagneticDipole(PointSource):

    m = JProperty(np.zeros(3))
    dm_dt = JProperty(np.zeros(3))
    d2m_dt2 = JProperty(np.zeros(3))

    def __init__(self, m=None, dm_dt=None, d2m_dt2=None, **kwargs):
        """
        :param location: callable or (..., 3) array, point source location.
        :param m: callable, uw dipole moment as a function of t.
        :param dm_dt: callable, first derivative of m as a function of t.
        :param d2m_dt2: callable, second derivative of m as a function of t.
        :param cut_off: float, exclusion distance around the source location.
        Avoids very large numbers near the singularity. The fields are nan if
        called within this region.
        """
        super().__init__(**kwargs)
        self.m = m
        self.dm_dt = dm_dt
        self.d2m_dt2 = d2m_dt2

    @classmethod
    def oscillator(cls, m=(0, 0, 1), freq=1e6, phase=0.0, **kwargs):
        """
        Sinusodially oscillating dipole.
        :param location: callable or (..., 3) array, point source location.
        :param freq: float, frequency in Hz.
        :param m: 3-array, dipole moment.
        :param phase: float, phase in radians.
        :param cut_off: float, exclusion distance around the source location.
        """
        #TODO change into power irradiated? or as option?
        m = np.array(m)
        osc = HarmonicOscillator(amplitude=m, freq=freq, phase=phase)
        return cls(m=osc.f, dm_dt=osc.df_dt, d2m_dt2=osc.d2f_dt2, **kwargs)

    def E_retarded_coordinates(self, R, tau):
        """ Computes the electric field from retarded coordinates. """
        R_norm, R_unit = vec_split(R)
        far = cross(R_unit, self.d2m_dt2(tau)) / (R_norm * Constants.C)
        near = cross(R_unit, self.dm_dt(tau)) / (R_norm ** 2)
        return Constants.K_e / Constants.C * (far + near)

    def B_retarded_coordinates(self, R, tau):
        """ Computes the uw field from retarded coordinates. """
        R_norm, R_unit = vec_split(R)
        far = (cross(R_unit, cross(R_unit, self.d2m_dt2(tau)))
               / (Constants.C ** 2 * R_norm))
        near1 = ((3 * dot_keepdim(R_unit, self.dm_dt(tau)) * R_unit
                  - self.dm_dt(tau)) / (Constants.C * R_norm ** 2))
        near2 = ((3 * dot_keepdim(R_unit, self.m(tau)) * R_unit
                  - self.m(tau)) / R_norm ** 3)
        return Constants.K_u * (far + near1 + near2)


class WireSegment(PointSource):

    #TODO d2I_dt2
    """
    A wire segment is a line that carries a current. It's assumed that a segment
    is small enough to be approximated by a points source.
    """

    line = JProperty()
    I = JProperty(1.0)
    dIdt = JProperty(0.0)

    def __init__(self, line=None, I=None, dI_dt=None, **kwargs):
        """
        It is the responsibility of the user to create sensible
        configurations. This mostly concerns charge conservation.
        :param line: callable or (2,3)-array, start and end coordinate.
        :param I: callback or float, current as a function of time.
        :param dI_dt: callback or float, derivative of the  current as a
                     function of time.
        :param cut_off: float, length around location for which nan's are
                        returned.
        """
        self.line = line  # set attribute before super().__init__()
        super().__init__(location=self._center, **kwargs)
        self.I = I
        self.dIdt = dI_dt

    def _center(self, t):
        """ Returns the center of the segment as function of time t."""
        r1, r2 = self.line(t)
        return (r1 + r2) / 2

    def _tangent(self, t):
        """ Returns the tangent of the segment as function of time t."""
        r1, r2 = self.line(t)
        return r1 - r2

    def E_retarded_coordinates(self, R, tau):
        R_norm = norm_keepdim(R)
        term1 = self.dIdt(tau) * self._tangent(tau) / R_norm
        return Constants.K_e / Constants.C ** 2 * term1

    def B_retarded_coordinates(self, R, tau):
        R_norm, R_unit = vec_split(R)
        term1 = self.I(tau) * self._tangent(tau) / R_norm ** 2
        term2 = self.dIdt(tau) * self._tangent(tau) / R_norm / Constants.C
        return Constants.K_u * cross(term1 + term2, R_unit)
        #TODO check this equation


class Wire(SourceCollection):
    """
    A collection of consecutive source_segments that carry the same current.
    A wire is a collection of wire source_segments. The segmentation is built-in if
    one of the class methods is used (from_curve(), polygon(), or cicrle())
    It is possible to define wires that are not a loop, but be careful
    with the interpretation as such a construct does not conserve charge.
    """

    line = JProperty()
    I = JProperty(1.0)
    dIdt = JProperty(0.0)
    movement = JProperty(np.zeros(3))

    @classmethod
    def curve(cls, curve, num_segments=50, I=None, dI_dt=None):
        """
        Creates a wire based on an analytical curve. It handles segmentation
        of the curve into PointSources.  A curve is some function
        of s and t, where s is the curve parametrization between 0 and 1,
        and t is the time. The curve function should return a 3D coordinate when
        called for s and t. Note that callbacks are returned for the source_segments,
        so the curve can change shape as a function of time.

        :param curve: callable function of s and t
        :param num_segments: int, number of PointSource elements to subdivide
                             he curve in. Higher is more accurate, but increases
                             computation time.
        :param I: callback or float, current as a function of time.
        :param dI_dt: callback or float, derivative of the  current as a
                     function of time.
        """
        s = np.linspace(0, 1.0, num_segments + 1)
        segments = []
        for i in range(0, num_segments):
            def line(t, i=i):
                return (curve(s[i], t), curve(s[i + 1], t))
            segments.append(WireSegment(line=line, I=I, dI_dt=dI_dt))
        return cls(segments)

    @classmethod
    def polygon(cls, nodes, num_segments=50, I=None, dI_dt=None, movement=None):
        """
        A wire based on coordinates connected by straight lines.
        :param nodes: (:, 3) array, coordinates of the polygon.
        :param num_segments: int, number of PointSource elements to subdivide
                             he curve in. Higher is more accurate, but increases
                             computation time.
        :param I: callback or float, current as a function of time.
        :param dI_dt: callback or float, derivative of the  current as a
                     function of time.
        :param movement: callable, function of time that returns a vector
                         which is added to all curve elements. This makes it
                         possible to let the polygon move in space as a rigid
                         body.
        """
        movement = default(movement, lambda t: np.zeros(3))
        nodes = np.array(nodes)
        l = [nodes[i + 1, :] - nodes[i, :] for i in range(len(nodes[:, 0]) - 1)]
        l_norm = np.array(list(map(np.linalg.norm, l)))
        L = np.sum(l_norm)
        l_sum = np.cumsum(np.concatenate(([0], l_norm)))

        def curve(s, t):
            i = np.argmax((s * L) < l_sum) - 1
            return movement(t) + nodes[i, :] \
                   + (s * L - l_sum[i]) * l[i] / np.linalg.norm(l[i])

        return cls.curve(curve, num_segments, I, dI_dt)

    @classmethod
    def circle(cls, center=(0, 0, 0), radius=1.0,
               normal=None, movement=None,
               num_segments=50, I=1.0, dI_dt=0.0):
        """
        Creates a circular wire.
        :param center: 3-array,
        :param radius: float,
        :param normal: 3-array, axial vector wrt the circle surface.
        :param num_segments: int, number of PointSource elements to subdivide
                             he curve in. Higher is more accurate, but increases
                             computation time.
        :param I: callback or float, current as a function of time.
        :param dI_dt: callback or float, derivative of the  current as a
                     function of time.
        :param movement: callable, function of time that returns a vector
                         which is added to all curve elements. This makes it
                         possible to let the circle move in space.
        """

        movement = default(movement, lambda t: np.zeros(3))
        normal = default(normal, (0, 0, 1))
        normal = normal / np.linalg.norm(normal)
        u = np.linalg.norm(np.array([-normal[0], normal[1], 0]))
        if normal[0] == 0 and normal[1] == 0:
            u = np.array([0, 1, 0])
        v = cross(normal, u)
        v = v / np.linalg.norm(v)
        def curve(s, t):
            return movement(t) + center + radius * u * np.cos(s * 2 * np.pi) \
                   + radius * v * np.sin(s * 2 * np.pi)
        return cls.curve(curve, num_segments, I, dI_dt)

    def get_segmentation(self, t=0):
        coordinates = []
        for segment in self.values():
            coordinates.extend(segment.line(t))
        return np.array(coordinates)


class Uniform(Source):
    """ Space independent fields. """

    _E = JProperty(np.zeros(3))
    _B = JProperty(np.zeros(3))

    def __init__(self, E=None, B=None):
        #super().__init__()
        self._E = E
        self._B = B

    @broadcast_spacetime
    def E(self, r, t):
        return match_shape(self._E(t), r)

    @broadcast_spacetime
    def B(self, r, t):
        return match_shape(self._B(t), r)



class PointCharge(PointSource):

    """ TODO Infinitely small electric  charge,
    The Liénard–Wiechert potential needs to be put here.
    """

    q = JProperty()

    def __init__(self, location=(0, 0, 0), q=None, cut_off=1.0):
        """
        """
        pass

    @classmethod
    def oscillator(cls):
        pass

    def E_retarded_coordinates(self, R, tau):
       pass

    def B_retarded_coordinates(self, R, tau):
        pass


class Observer:
    """"
    A set of registered measurement locations that observes a collection of
    sources. A convenient way to use jefpy. Yields E and B as functions of t.
    These functions can be used directly or used as call-backs in visualizations.
    """

    r = JProperty()

    def __init__(self, r, sources, observable='uvw', transform=None):
        """
        :param r: (..., 3)-array, callback: observer location
        :param sources: collection of Source objects.
        """
        if not isinstance(sources, SourceCollection):
            if isinstance(sources, Source):
                sources = {'source': sources}
            sources = SourceCollection(sources)
        self.sources = sources
        self.sources.observable = observable
        self.sources.transform = transform or (lambda r, F: F)
        self.r = r

    def E(self, t=0.0):
        return self.sources.E(self.r(t), t)

    def B(self, t=0.0):
        return self.sources.B(self.r(t), t)

    def S(self, t=0.0):
        return self.sources.S(self.r(t), t)

