"""
Potentials
==========

Potential energy functions.

.. autosummary::
    :nosignatures:

    LennardJones

.. autoclass:: LennardJones
    :members:

"""
import numpy as np

class LennardJones:
    r"""Lennard-Jones pair potential.

    The prototypical pair potential consisting of a steep repulsive core
    and an attractive tail describing dispersion forces. The functional form
    of the potential is:

    .. math::

        u(r) = \begin{cases}
               4 \varepsilon\left[\left(\dfrac{\sigma}{r}\right)^{12}
               - \left(\dfrac{\sigma}{r}\right)^6 \right], & r \le r_{\rm cut} \\
               0, & r > r_{\rm cut}
               \end{cases}

    where :math:`r` is the distance between the centers of two particles,
    :math:`\varepsilon` sets the strength of the attraction, and
    :math:`\sigma` sets the length scale of the interaction. (Typically,
    :math:`\sigma` can be regarded as a particle diameter.) The potential
    is truncated to zero at :math:`r_{\rm cut}`, and good accuracy for
    thermodynamic properties is usually achieved when :math:`r_{\rm cut} \ge 3\sigma`.

    In molecular dynamics (MD) simulations, the forces on the particles are what
    is actually required. Forces are computed from the derivative of :math:`u(r)`
    with the truncation scheme:

    .. math::

        \mathbf{F}(\mathbf{r}) = \begin{cases}
                                 f(r) \mathbf{r}/r, & r \le r_{\rm cut} \\
                                 0, & r > r_{\rm cut}
                                 \end{cases}

    where :math:`f(r) = -\partial u/\partial r`. The force is a vector with
    direction. If :math:`\mathbf{r}` is the vector pointing from a particle *i*
    to a particle *j*, then the force on *j* is :math:`\mathbf{F}` and the force
    on *i* is :math:`-\mathbf{F}`.

    This force truncation implies that the energy should be shifted to zero at
    :math:`r_{\rm cut}` by subtracting :math:`u(r_{\rm cut})`. However, this
    distinction is often not made in MD simulations unless thermodynamic properties
    based on the energy are being computed. Caution must be taken if MD results
    with this scheme are compared to Monte Carlo (MC) results, which are sensitive
    to whether :math:`u` is shifted or not.

    If the Lennard-Jones potential is truncated and shifted at its minimum
    :math:`r_{\rm cut} = 2^{1/6}\sigma`, the interactions are purely repulsive.
    (The forces are always positive, :math:`|\mathbf{F}| \ge 0`.)
    This special case is often used as an approximation of nearly hard spheres,
    where it is referred to as the Weeks--Chandler--Andersen potential based
    on its role in their perturbation theory of the liquid state.

    Parameters
    ----------
    epsilon : float
        Interaction energy.
    sigma : float
        Interaction length.
    rcut : float
        Truncation distance.
    shift : bool
        If ``True``, shift the potential to zero at ``rcut``.

    """
    def __init__(self, epsilon, sigma, rcut, shift=False):
        self.epsilon = epsilon
        self.sigma = sigma
        self.rcut = rcut
        self.shift = shift

    def compute(self, state):
        r"""Compute energy and forces on particles.

        The pair potential is evaluated using a direct calculation between
        all :math:`N^2` pairs in the ``state``. Half of the potential energy is
        assigned to each particle in the pair.

        Hint: the pair calculation can be efficiently implemented using NumPy arrays::

            for i in range(state.N-1):
                # get dr with all j particles ahead of myself (count each pair once)
                drij = state.positions[i+1:]-state.positions[i]

                # do the calculations to get uij and fij from drij
                # ...

                # accumulate the result
                u[i] += np.sum(uij)
                u[i+1:] += uij
                f[i] -= np.sum(fij,axis=0)
                f[i+1:] += fij

        Parameters
        ----------
        state : :class:`~learnmolsim.state.State`
            Simulation state.

        Returns
        -------
        :class:`numpy.ndarray`
            Potential energy assigned to each particle.
        :class:`numpy.ndarray`
            Force on each particle.

        """
        raise NotImplementedError()

    def energy_force(self, rsq):
        r"""Evaluate potential energy and force magnitude.

        Efficiently implements the functional form of the potential. Accepting
        :math:`r^2` rather than *r* means no square root needs to be evaluated.
        The potential energy :math:`u(r)` and the force divided by *r*,
        i.e., :math:`f(r)/r`, are evaluated directly using :math:`r^2`. Factoring
        the :math:`1/r` here means that the force vector can be applied without
        normalization:

        .. math::

            \mathbf{F}(\mathbf{r}) = \frac{f(r)}{r} \mathbf{r}

        If any ``rsq`` is 0, the energy and force is :py:obj:`numpy.inf`.

        The return type will be a scalar or array depending on the type of ``rsq``.

        Parameters
        ----------
        rsq : float or array_like
            Squared pair distance.

        Returns
        -------
        float or :class:`numpy.ndarray`
            Energy at the pair distances.
        float or :class:`numpy.ndarray`
            Force divided

        """
        raise NotImplementedError()

    @classmethod
    def _zeros(cls, x):
        """Ensure a 1d NumPy array of zeros to match coordinates.

        This function can be used to ensure coordinates are consistently treated
        as a 1d array. If the coordinate ``x`` is a scalar (a float), it is
        promoted to a one-element NumPy array. If ``x`` is a 1d array, nothing is
        done. Higher dimensional arrays are rejected.

        The shape of the returned array matches the shape of ``x`` as an array.

        A flag is returned to indicate if ``x`` was originally a scalar. This can
        be used to downconvert to the same input type::

            x,f,s = self._zeros(x)
            # do something
            if s:
                f = f.item()
            return f

        Parameters
        ----------
        x : float or array_like
            Coordinates to make an array of zeros for.

        Returns
        -------
        :class:`numpy.ndarray`
            Coordinates promoted to a NumPy array.
        :class:`numpy.ndarray`
            Empty array matching the returned coordinates.
        bool
            True if coordinates were originally a scalar quantity.

        Raises
        ------
        TypeError
            If the coordinates are not castable to a 1d array.

        """
        s = np.isscalar(x)
        x = np.array(x, dtype=np.float64, ndmin=1)
        if len(x.shape) != 1:
            raise TypeError('Coordinate must be scalar or 1D array.')
        return x,np.zeros_like(x),s
