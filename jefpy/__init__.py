"""

* Tag-line: Closed-form electromagnetic solver
* Author: Leon van Kouwen, lvankouwen@gmail.com 
* Last updated: 2021-02-27


# About jefpy

jefpy is a package for simulating electromagnetic fields. The
computations are based on Jefimenko's equations, which are
essentially the integral form of Maxwell equations, evaluated
at retarded time.

A big advantage over FEM or FDTD simulations is that no mesh is
required. In addition, when only a few sample points are required,
it can be much faster. The big limitation, however, is that material
properties can't be accounted for (at least currently not). In
essence it is a free space simulator with sources.

Despite this limitation, this package can be very convenient to test
simple systems. Simulations are easy to set up, fast, and standard
visualization shows one quickly what is going on.

For low frequencies the results are equal to the quasi-static use of
the Coulomb law and the Biot-Savart law. For higher frequencies, when the
quasi-static approach breaks down, this packages remains valid.

This package is just released. Please help make it more mature! This can be done
by filing issues, request and suggestions here on github. If would also be happy
to hear what you are using it for.


# How do I get set up?

## Installation
Navigate to the package folder and use the following command to install:

    pip install .

## Dependencies

Tested using:

* python==3.8.0
* numpy==1.18.1
* matplotlib==3.3.2 (only if visualizations are used)
* pytest==6.1.1 (for unit tests)


# API documentation

## Examples
To get started with some examples look at the Jupyter Notebooks in the "notebooks\" folder.

## Main workflow
The basic procedure of using the package is as follows:

* Create one or more sources
* Combine multiple sources in a SourceCollection
* Use the methods .E(r, t=0.0) and .B(r, t=0.0) to calculate fields.

When using the visualization tools of the package:

* Define the observation location set (points, surface, ...)
* Combine the SourceColletion and the observation locations in a Observer object
* Pass the .E or .B callbacks to the visualization objects.

## Dimensions
By convention the last dimension is always used for the directional
components (x,y,z) or (r, theta, phi), (Ex, Ey, Ez) etc. So all
vectorfields have shape signature [..., 3] when time series are used,
the first dimension is used, giving the signature [Nt, ..., 3]
The package can handle any number of dimensions in between. Some examples:

- Electric field sampled at a point, for different times: [Nt, 3]
- Electric field sampled at a line, for different times: [Nt, Nu, 3]
- Electric field sampled on a surface: [Nu, Nv, 3]
- Electric field sampled in a volume, for different times: [Nt, Nu, Nv, Nw, 3]
- Electric field sampled in a set of volumes, for different times: [Nt, Nset, Nu, Nv, Nw, 3]

## Polymorphism of properties
Most properties are time-dependent callbacks, but can be set as static
vectors or values. Note that internally when passing a static value
to a source constructor, a callback is generated which returns the static
value for any t.

For example the location property can be set by passing a function reference
with a time argument which outputs the location vector. Alternatively
one can input a vector directly and rely on the internal jefpy mechanics to
make it a callback.

## SourceCollection
Source can be combined by using the SourceCollection class. This
class inherits from dict, and can be used as such. In addition it has
the methods E, B and S. These methods give the combined fields of all
sources in the collection. Note that nesting of source collections is
possible, allowing very flexible grouping constructions of sources.
Source collections can be created by using the SourceCollection consturctor directly,
but the '+' for Source and SourceCollection is also functional yielding
the following SourceCollection arithmetic:

- Source + Source = new SourceCollection
- Source + SourceCollection = original SourceCollection object,
with the Source object included.
- SourceCollection + SourceCollection = left side SourceCollection,
with all the sources from the second SourceCollection added.

## Observers
It can be convenient to pre-define the location(s) on which to evaluate
the field ons. The Observer class manages this by yielding the methods
E(t), B(t), S(t). Note that these are only time-dependent.

## Notes
* All units are SI.
* Currently the permittivity and permeability of free space are fixed
for the module and set in "physics.Constants".
* the time parameter is by default always 0.0. This can be used for static
configurations by omitting the t argument, i.e. .E(r) and .B(r).

See the docs folder for additional information. The docstrings do not preserve
linebreaks so consider using "EXPAND SOURCE CODE" when inspecting a method.


## Developers

See the test folder for the available tests. To run all tests, install
pytest and run from the jefpy folder:

    pytest tests

- When creating new sources, inherit from Source, PointSource or SourceCollection.
- For point sources, overwrite `E_retared_coordinates` and/or `B_retared_coordinates`
by assuming the source is located at the origin. The retarded coordinates functionality
handle the positioning in space.
- To achieve polymorphism of properties the JProperty class is used extensively. Note that
it manipulates the set operator ("=") of attributes that are defined as class attributes.
Check out the code in  the utils module.

# Possible Future Work
- Implement a dynamic electric point source (Liénard–Wiechert potential)
- Implement antenna's
- Support for simple uw materials
- Support for simple dielectric materials
- Support reflective planes
"""


# The package is small enough to load everything unnested.

from .physics import *
from .visualize import *
from .math import *
from .geometry import *
from .utils import *


