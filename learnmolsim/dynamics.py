"""
Dynamics
========

Methods for dynamically evolving a :class:`~learnmolsim.state.State`.

.. autosummary::
    :nosignatures:

    VelocityVerlet

.. autoclass:: VelocityVerlet
    :members:

"""

class VelocityVerlet:
    r"""Velocity Verlet integration.

    This integration method implements the classical NVE molecular dynamics
    timestepper. The algorithm is:

    .. math::

        &\mathbf{v}_i(t+\Delta t/2) = \mathbf{v}_i(t) + \frac{\Delta t}{2 m_i} \mathbf{f}_i(t)

        &\mathbf{r}_i(t+\Delta t) = \mathbf{r}_i(t) + \Delta t \mathbf{v}_i(t+\Delta t/2)

        &\mathbf{v}_i(t+\Delta) = \mathbf{v}_i(t+\Delta/2) + \frac{\Delta t}{2 m_i} \mathbf{f}_i(t+\Delta t)

    where the forces :math:`\mathbf{f}` are determined from the positions
    :math:`\mathbf{r}` at the indicated timestep.

    Arguments
    ---------
    dt : float
        Integration time step :math:`\Delta t`.
    potential : :class:`~learnmolsim.potential.LennardJones`
        Interaction potential. (Currently, only one is implemented.)

    """
    def __init__(self, dt, potential):
        self.dt = dt
        self.potential = potential

    @property
    def dt(self):
        """float: Integration time step."""
        return self._dt

    @dt.setter
    def dt(self,value):
        if value < 0:
            raise ValueError('Timestep must be nonnegative')
        self._dt = value

    def advance(self, state):
        """Advance the simulation state.

        Take one step of the velocity Verlet algorithm. The evaluated energies
        and forces are stored in the :class:`~learnmolsim.state.State`, and the ``State.counter``
        is advanced by one. The positions are forced back into the :class:`~learnmolsim.state.Box`
        using :meth:`~learnmolsim.state.Box.wrap`.

        When starting the algorithm, the velocities or forces in the :class:`~learnmolsim.state.State`
        may not be set. The velocities can be defaulted to zero. The forces need
        to be initially computed (for the first step of the update), and so they
        may need to be computed an extra time.

        """
        raise NotImplementedError()
