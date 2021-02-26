"""
State
=====

Data structures for defining the state of the simulation.

.. autosummary::
    :nosignatures:

    Box
    State

.. autoclass:: Box
    :members:

.. autoclass:: State
    :members:

"""
import numpy as np

class Box:
    r"""Orthorhombic simulation box with periodic boundary conditions.

    An orthorhombic box is a rectangular prism characterized by three edge
    lengths :math:`\mathbf{L} = (L_x,L_y,L_z)`. If only one length ``L`` is
    specified, the box is treated as a cube. The origin of the box is assumed
    to be :math:`(0,0,0)` throughout.

    The :class:`Box` defines the geometry of the simulation. It also helps
    enforce the periodic boundary conditions through the :meth:`wrap` and
    :meth:`minimum_image` methods. All particle coordinates should like between
    **0** and **L**.

    Typically, a :class:`Box` will be constructed and associated with a
    simulation :class:`State`.

    Parameters
    ----------
    L : float or array_like
        Edge lengths.

    Examples
    --------
    Create a cube::

        box = learnmolsim.state.Box(10.0)

    Create an orthorhombic box::

        box = learnmolsim.state.Box((10.0,15.0,20.0))

    """
    def __init__(self, L):
        self.L = L

    @property
    def L(self):
        """:class:`numpy.ndarray`: Edge lengths."""
        return self._L

    @L.setter
    def L(self, value):
        L = np.array(value, ndmin=1, dtype=np.float64)
        if L.shape == (1,):
            L = np.repeat(L,3)

        if L.shape != (3,):
            raise TypeError('Box size must be a 3-element array')

        if np.any(L <= 0):
            raise ValueError('Box size must be positive')

        self._L = L

    def volume(self):
        """Compute volume of the box.

        Returns
        -------
        float
            The volume of the box.

        """
        raise NotImplementedError()

    def wrap(self, position, image=None):
        """Wrap particles into the box.

        When a particle exits the :class:`Box`, it needs to be "wrapped" back
        in because of the periodic boundary condition. That is, a particle
        exiting one face of the box should reenter from the opposite face.
        Mathematically, this wrapping can be achieved using :func:`numpy.floor` to
        compute the integer number of "images" the particle has moved beyond the
        current box. These images can then be subtracted from the current
        position::

            r_wrap = r - L*num_image

        In order to be able to reconstruct the unwrapped particle coordinates,
        this shift in image can be tracked by supplying the optional ``image``
        argument. The images are updated as::

            image += num_image

        and the unwrapped coordinates are::

            r_unwrap = r_wrap + L*image

        This method can wrap either a single particle position as a 3-element
        1d array, or many particle positions as a 2d array. For efficiency,
        this method will attempt **not** to copy the ``position`` and ``image``
        arrays. Hence, they will be modified in place and returned if
        :func:`numpy.array` does not return a copy; otherwise, a copy will be
        returned.

        Parameters
        ----------
        position : array_like
            Particle position(s). The array should either have shape ``(3,)``
            or ``(N,3)``, and the data type should be castable to ``numpy.float64``.
        image : array_like
            Particle image(s). Optional, defaults to ``None``. The array
            shape should match the ``position``, and the data type
            should be castable to ``numpy.int32``.

        Returns
        -------
        :class:`numpy.ndarray`
            Wrapped particle positions.
        :class:`numpy.ndarray`
            Wrapped particle images, if ``image`` was provided.

        """
        raise NotImplementedError()

    def minimum_image(self, vector):
        """Compute the minimum image of a vector.

        Apply the minimum image convention to a vector between two particles
        so that it always points the shortest route between them.

        The minimum-image vector within the periodic boundaries can be computed
        using :func:`numpy.round` to remove the appropriate number of images::

            v -= round(v/L)*L

        Arguments
        ---------
        vector : array_like
            Particle position(s). The array should either have shape ``(3,)``
            or ``(N,3)``, and the data type should be castable to ``numpy.float64``.

        Returns
        -------
        :class:`numpy.ndarray`
            Minimum image of ``vector``.

        """
        raise NotImplementedError()

class State:
    """Simulation state.

    The information needed to describe the state of the particles is tracked
    in one convenient data structure. The data are laid out using a "structure
    of arrays," where each particle property is stored in a separate array. This
    turns out to be computationally convenient.

    All ``N`` particles in the state share a common ``box`` and ``mass``. The
    logical state of the system is tracked using a ``counter``. In molecular
    dynamics simulations, this would represent the integration time step, but in
    other techniques, it might represent some other tracker.

    The information tracked for all particles are the :attr:`positions` and
    :attr:`images` (see :meth:`Box.wrap` for the meaning of an image). Additionally,
    the :attr:`velocities`, :attr:`energies`, and :attr:`forces` can optionally
    be set, but they default to ``None`` initially.

    Parameters
    ----------
    N : int
        Number of particles
    box : :class:`Box`
        Simulation box.
    mass : float
        Mass of all particles (default: 1.0).
    counter : int
        Initial counter (default: 0).

    Example
    -------
    Create a new state with 20 particles of mass 5::

        box = learnmolsim.state.Box(10.0)
        state = learnmolsim.state.State(20, box, mass=5.0)
        state.positions = box.L*numpy.random.uniform(size=(state.N,3))
        state.velocities = numpy.zeros((state.N,3))

    """
    def __init__(self, N, box, mass=1.0, counter=0):
        if N < 0 or not isinstance(N,int):
            raise ValueError('Number of particles must be nonnegative integer')
        self._N = N

        self.box = box
        self.mass = mass
        self.counter = counter

        self.positions = np.zeros((self.N,3),dtype=np.float64)
        self.images = np.zeros((self.N,3),dtype=np.int32)
        self.velocities = None
        self.energies = None
        self.forces = None

    @property
    def N(self):
        """int: Number of particles."""
        return self._N

    @property
    def box(self):
        """:class:`Box`: Simulation box."""
        return self._box

    @box.setter
    def box(self, value):
        if not isinstance(value, Box):
            raise TypeError('box must be a Box object')
        self._box = value

    @property
    def mass(self):
        """float: Particle mass."""
        return self._mass

    @mass.setter
    def mass(self, value):
        if value <= 0.0:
            raise ValueError('Mass must be positive')
        self._mass = value

    @property
    def counter(self):
        """int: Counter."""
        return self._counter

    @counter.setter
    def counter(self, value):
        if not isinstance(value, int):
            raise TypeError('Counter must be an integer')
        self._counter = value

    @property
    def positions(self):
        """:class:`numpy.ndarray`: Particle positions (``numpy.float64``)."""
        return self._positions

    @positions.setter
    def positions(self, value):
        r = np.array(value, ndmin=2, copy=False, dtype=np.float64)
        if r.shape != (self.N,3):
            raise TypeError('Positions must be an Nx3 array')
        self._positions = r

    @property
    def images(self):
        """:class:`numpy.ndarray`: Particle images (``numpy.int32``)."""
        return self._images

    @images.setter
    def images(self, value):
        im = np.array(value, ndmin=2, copy=False, dtype=np.float64)
        if im.shape != (self.N,3):
            raise TypeError('Images must be an Nx3 array')
        self._images = im

    @property
    def velocities(self):
        """:class:`numpy.ndarray`: Particle velocities (``numpy.float64``)."""
        return self._velocities

    @velocities.setter
    def velocities(self, value):
        if value is None:
            self._velocities = value
        else:
            v = np.array(value, ndmin=2, copy=False, dtype=np.float64)
            if v.shape != (self.N,3):
                raise TypeError('Velocities must be an Nx3 array')
            self._velocities = v

    @property
    def energies(self):
        """:class:`numpy.ndarray`: Particle energies (``numpy.float64``)."""
        return self._energies

    @energies.setter
    def energies(self, value):
        if value is None:
            self._energies = None
        else:
            e = np.array(value, ndmin=1, copy=False, dtype=np.float64)
            if e.shape != (self.N,):
                raise TypeError('Energies must be an N array')
            self._energies = e

    @property
    def forces(self):
        """:class:`numpy.ndarray`: Particle forces (``numpy.float64``)."""
        return self._forces

    @forces.setter
    def forces(self, value):
        if value is None:
            self._forces = None
        else:
            f = np.array(value, ndmin=2, copy=False, dtype=np.float64)
            if f.shape != (self.N,3):
                raise TypeError('Forces must be an Nx3 array')
            self._forces = f
