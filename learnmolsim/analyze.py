"""
Analysis tools
==============

Tools for analyzing the :class:`~learnmolsim.state.State` of a simulation.

.. autosummary::
    :nosignatures:

    Thermodynamics

.. autoclass:: Thermodynamics
    :members:

"""
import numpy as np

class Thermodynamics:
    """Compute thermodynamic properties of a state."""

    def kinetic_energy(self, state):
        r"""Compute the kinetic energy.

        The kinetic energy :math:`E_{\rm k}` of *N* particles is:

        .. math::

            E_{\rm k} = \sum_{i=1}^N \frac{1}{2} m_i \mathbf{v}_i \cdot \mathbf{v}_i

        where *m* is the mass of particle *i* and :math:`\mathbf{v}_i` is the
        velocity vector of particle *i*.

        Parameters
        ----------
        state : :class:`~learnmolsim.state.State`
            Simulation state.

        Returns
        -------
        float
            Kinetic energy of the ``state``.

        """
        raise NotImplementedError()

    def potential_energy(self, state):
        r"""Compute the potential energy.

        Contributions of potential energies :math:`U` are arbitrarily
        assigned to each particle as :math:`U_i`. For example, half of the
        potential energy is assigned to each particle interacting through a
        pair potential. The total potential energy is the sum of these contributions:

        .. math::

            U = \sum_{i=1}^N U_i

        Parameters
        ----------
        state : :class:`~learnmolsim.state.State`
            Simulation state.

        Returns
        -------
        float
            Potential energy of the ``state``.

        """
        raise NotImplementedError()

    def kT(self, state):
        r"""Compute the thermal energy.

        The temperature of *N* particles is computed from the kinetic energy using
        the equipartition theorem:

        .. math::

            k_{\rm B} T = \frac{2}{3 N} E_{\rm k}

        Note that some codes will remove the center-of-mass velocity of the
        particles when computing the temperature. This constrains 3 degrees of freedom
        in the system, and the factor in the denominator becomes :math:`3(N-1)`.
        This factor can be further reduced if additional constraints on particles
        are present.

        Parameters
        ----------
        state : :class:`~learnmolsim.state.State`
            Simulation state.

        Returns
        -------
        float
            Thermal energy (:math:`k_{\rm B}T`) of the ``state``.

        """
        raise NotImplementedError()

    def pressure(self, state):
        r"""Compute the pressure.

        The pressure of *N* particles in a volume *V* is given by the virial
        theorem:

        .. math::

            P = \frac{N k_{\rm B} T}{V} + \frac{1}{3V} \sum_{i=1}^N \mathbf{r}_i \cdot F_i

        where :math:`\mathbf{r}_i` is the position of particle *i* and
        :math:`\mathbf{F}_i` is the force on particle *i*.

        Parameters
        ----------
        state : :class:`~learnmolsim.state.State`
            Simulation state.

        Returns
        -------
        float
            Pressure of the ``state``.

        """
        raise NotImplementedError()
