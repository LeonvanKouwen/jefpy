
""" 
Collection of useful visualization options for vector fields.
Matches well with the outputs from the physics.py module.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d
from typing import Callable, Iterable
from abc import ABC, abstractmethod
from collections import deque
import time



class Movie(ABC):
    """ Abstract base class for dynamic visualizations of fields. """

    def __init__(self, space, field):
        """
        Creates a movie object. A fig object is created such that figure options can be
        set after object creation by using obj.fig and obj.ax.
        :param space: (..., 3) array, coordinate base for plotting.
        :param field: callable, function of t that returns values to plot versus
                      the coordinate base (space).
        """
        self.space = space
        self.field = field

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('close_event', self.close_fig)
        self.is_closed = False
        self.fig.tight_layout()
        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.plot = None
        self.range = None
        self.title = ""


    @abstractmethod
    def snapshot(self, t):
        """
        Method that plots data for a particular time. Should create the plotting
        object such that it can be called independently.
        :param t: float, time.
        """
        return

    def snapshot_update(self, t):
        """
        Updates the data in every frame. The difference with snapshot is that this method
        does not need to, and should not, recreate the plotting object. For example use
        set_data() in stead of plot(). Should be implemented if the 'live' or 'record' is used.
        :param t: float, time.
        """
        raise NotImplementedError

    def live(self, t_max=1.0e99, t0=0.0, slow_motion=1e-9, run_time=20):
        """
        Start plotting with live dynamic updates. This is a blocking operation. Note that
        it can't be (easily) put in a separate thread as many visualizations require to be
        running in the main thread.
        :param t_max: float, Efield_queue end time.
        :param t0: float, Efield_queue start time.
        :param slow_motion: float, amount of slowing down Efield_queue time to real time
        :param run_time: float, amount of real time the plotting is live.
        """
        plt.ion()
        plt.show()
        self.is_closed = False
        run_time_0 = time.time()
        t = t0
        self.snapshot(t)
        while t < t_max and (time.time() - run_time_0) < run_time:
            t = t0 + slow_motion * (time.time() - run_time_0)
            self.snapshot_update(t)
            #plt.draw()
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
            if self.is_closed:
                break
        plt.close(self.fig)



    def close_fig(self, evt):
        self.is_closed = True

    def record(self, filename='movie.mp4', sim_time=1e-9, slowmotion=1e-9,
               t0=0.0, FPS=30, dpi=300):
        """
        Save a movie to disk.
        :param filename: str, title of movie. Other extensions that .mp4 not tested.
        :param sim_time: float, total time for the Efield_queue to run.
        :param slowmotion: float, conversion factor from sim time to real time.
        :param t0: float, start of the Efield_queue.
        :param FPS: float, frames per second.
        :param dpi: float, movie resolution.
        :return:
        """
        dt = slowmotion / FPS
        N = round(sim_time / dt)
        t = np.linspace(t0, t0 + sim_time - dt, N)
        self.snapshot(t0)

        def frame(ti):
            readiness = int(ti / sim_time * 100)
            print(f"Recording movie: {readiness} % ready", flush=True, end="\r")
            self.snapshot_update(ti)

        animation = FuncAnimation(self.fig, frame, frames=t, interval=1000.0/FPS)
        animation.save(filename, dpi=dpi)
        plt.close(self.fig)


class MovieMap(Movie):

    """
    Create dynamic visualizations of fields in a map format. A map is spatially 2D,
    spanning a 1D heat map. The heat map is often the norm, or one component of the
    field.
    """

    def __init__(self, field):
        """
        Object for creating movies of field maps. Initializes the fig object
        and saves some settings. The figure options can be set after object
        creation by using obj.fig and obj.ax. Settings passed to imshow are accesed
        by obj.settings. Use obj.range to set the 'extend' argument of imshow.
        :param space: (..., 3) array, coordinate base for plotting.
        :param field: callable, function of t that returns values to plot versus
                      the coordinate base (space).
        """
        super().__init__(None, field)
        self.settings = {
            'interpolation': 'bilinear',
            'origin': 'lower',
            'cmap': "RdBu"}

    def snapshot(self, t):
        """
        Plots the field map for a particular time.
        :param t: float, time.
        """
        self.ax.set_title(f"{self.title}  t={t:.2e} s")
        self.plot = plt.imshow(self.field(t).T, extent=self.range,
                               **self.settings)

    def snapshot_update(self, t):
        """
        Updates the field map for a particular time. .snapshot should be called before.
        :param t: float, time.
        """
        self.plot.set_data(self.field(t).T)
        self.ax.set_title(f"{self.title}  t={t:.2e} s")



class MovieFlux(Movie):
    """ Create dynamic visualizations of 2D vector fields (quiver). """

    def __init__(self, space, field):
        """
        Object for creating 2D vector field plots (quiver). Initializes the fig object
        and saves some settings. The figure options can be set after object
        creation by using obj.fig and obj.ax.
        :param space: (..., 3) array, coordinate base for plotting.
        :param field: callable, function of t that returns values to plot versus
                      the coordinate base (space).
        """
        super().__init__(space, field)
        # Nothing other that the base init is executed. The init is here to
        # give specific documentation for this derived class.


    def snapshot(self, t):
        """
        Plots the vector field for a particular time.
        :param t: float, time.
        """
        self.ax.set_title(f"{self.title}  t={t:.2e} s")
        field = self.field(t)
        Fu, Fv = field.T
        F = np.sqrt(Fu**2 + Fv **2)
        args = (*self.space.T, Fu / F, Fv / F)
        self.plot = self.ax.quiver(*args)
        plt.show()

    def snapshot_update(self, t):
        """
        Updates the vector field for a particular time. .snapshot should be called before.
        :param t: float, time.
        """
        self.plot.set_UVC(*self.field(t).T)
        self.ax.set_title(f"{self.title}  t={t:.2e} s")


class MovieFluxEB(Movie):
    """ Special version of MovieFlux that plots E and B vectors simultaneously"""

    def __init__(self, space, E, B):
        """
        Object for creating 2D vector field plots (quiver) of E and B simultaneously.
        Initializes the fig object and saves some settings. The figure options can
        be set after object creation by using obj.fig and obj.ax.
        :param space: (..., 3) array, coordinate base for plotting.
        :param E: callable, Electric field as function of t.
        :param B: callable, Magnetic field as function of t.
        """
        # passing E to the parent class is just a dummy argument as self.field is not used.
        super().__init__(space, E)
        self.E = E
        self.B = B

    def snapshot(self, t):
        """
        Plots the vector fields for a particular time.
        :param t: float, time.
        """
        self.ax.set_title(f"{self.title}  t={t:.2e} s")
        self.plot_E = self.ax.quiver(*self.space.T, *self.E(t).T, color='r')
        self.plot_B = self.ax.quiver(*self.space.T, *self.B(t).T, color='b')
        plt.show()

    def snapshot_update(self, t):
        """
        Updates the vector fields for a particular time. .snapshot should be called before.
        :param t: float, time.
        """
        self.plot_E.set_UVC(*self.E(t).T)
        self.plot_B.set_UVC(*self.B(t).T)
        self.ax.set_title(f"{self.title}  t={t:.2e} s")



class Movie3Axes(Movie):
    """
    Create dynamic visualizations of observer points or sets of observer points
    as separate lines in three different subplots. Very similiar to "TimeSeries,
    only this object is dynamically updating the graphs.
    """

    def __init__(self, field, num_data_visible=100):
        """
        Object for creating observer point curves.  Initializes the fig object
        and saves some settings. The figure options can be set after object
        creation by using obj.fig and obj.ax. Note that this class is space location
        unaware.
        :param field: callable, function of t that returns values to plot.
        :param num_data_visible: int, maximum data point visible simultaneously in one frame.
        """
        super().__init__(None, field)
        plt.close(self.fig)
        self.fig, self.ax = plt.subplots(3, 1, sharex='all')
        self.t = deque(maxlen=num_data_visible)
        self.lines = np.empty(field(0.0).shape, dtype=object)
        self.field_buffer = np.empty(field(0.0).shape, dtype=object)
        for idx, _ in np.ndenumerate(self.field_buffer):
            self.field_buffer[idx] = deque(maxlen=num_data_visible)

    def snapshot(self, t):
        """
        Plots the field values for a particular time.
        :param t: float, time.
        """
        for idx, _ in np.ndenumerate(self.field_buffer):
            self.lines[idx], = self.ax[idx[-1]].plot(t, self.field(t)[idx])


    def snapshot_update(self, t):
        """
        Updates the field values for a particular time. .snapshot should be called before.
        :param t: float, time.
        """
        self.t.append(t)
        for idx, _ in np.ndenumerate(self.field_buffer):
            self.field_buffer[idx].append(self.field(t)[idx])
            self.lines[idx].set_data(self.t, self.field_buffer[idx])
            self.ax[idx[-1]].relim()
            self.ax[idx[-1]].autoscale_view(True, True, True)


class TimeSeries:
    """
    Plots the components of a three element vector in three subplots as a function of time.

    Can be used as a function because of the __call__ interface. The advantage over just a
    function is that various objects that may be changed are saved in this object (e.g. .fig, .ax)
    """

    def __init__(self, field):
        self.fig, self.ax = plt.subplots(3, 1, sharex='all')
        self.field = field
        self.lines = np.empty(field(0.0).shape, dtype=object)
        self.ax[0].set_ylabel('x')
        self.ax[1].set_ylabel('y')
        self.ax[2].set_ylabel('z')
        self.ax[2].set_xlabel('t (s)')

    def __call__(self, t):
        F = self.field(t)
        for idx, _ in np.ndenumerate(F[0, ...]):
            tricky_slice = tuple([slice(None)] + list(idx))
            self.lines[idx], = self.ax[idx[-1]].plot(t, F[tricky_slice])


def inspect_segments(segmentation, t=0):
    """
    Plots line source_segments in 3D. Specifically tailored to the "Wire" class of the physics module.
    The combination of source_segments are assumed to form a continuous curve.

    :param source_segments: objectcartesian
        .location: callable, returns a  coordinate as a function of time that describes
                   the center location of the segment.
        .direction: callable, returns a cartesian vector. The length of the vector is the length
                    of the segment. The direction of the vector described the orientation of the segment.
    :param t: float, time.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(*segmentation.T, 'o-')
    plt.show()


if __name__ == '__main__':
    pass