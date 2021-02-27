import cProfile
import jefpy as jp
import numpy as np

import jefpy.geometry

pr = cProfile.Profile()
pr.enable()

# power = 1.0
# direction = [0, 0, 1]
# frequency = 200e6
# location = [0, 0, 0]
# dipole1 = jp.ElectricDipole.oscillator(location, power, frequency, direction)
# field = jp.SourceCollection(sources=[dipole1])
#
# xrange, zrange, y, resolution = [0, 1], [-1, 1], 0.0, 0.01
# surface = jefpy.sampling.SamplingSurface.cartesian('xz', xrange, zrange, y, resolution)
# observer = jp.Observer(surface['XYZ'], field, observable='norm')
#
# for t in np.linspace(0, 1.0/frequency, 1000):
#     #observer.E(t)
#     observer.B(t)
#     #observer.E(t)




# import jefpy as jp
# import numpy as np
#
# dipole_settings = {
#     'power': 1.0,
#     'direction': [0, 0, 1],
#     'frequency': 200e6,
#     'location': [0, 0, 0]}
# dipole = jp.ElectricDipole.oscillator(**dipole_settings)
#
# r1 = [2.0, 2, 2]
# r2 = [-2.0, -2, -2]
# R = (r1, r2)
# observer = jp.Observer(R, dipole)
#
# visual = jp.Plot3axes(R, observer.B, num_data_visible=20)
#
# T = 1.0 / 200e6
# visual.record(sim_time=2*T)

pr.disable()
pr.dump_stats('tests\profile.prof')



