"""
Writers
=======

Writers to save the state of the system to a file.

.. autosummary::
    :nosignatures:

    XYZWriter

.. autoclass:: XYZWriter
    :members:

"""
class XYZWriter:
    """Write state to an extended XYZ file.

    The extended XYZ file format is a flexible but inefficient way to store the
    state of the system. The format is a sequence of frames written as plain
    text to a file with an ``.xyz`` extension. An example frame for a cubic
    box is::

        N
        Lattice="Lx 0 0 0 Ly 0 0 0 Lz" Time=counter
        type0 x0 y0 z0
        type1 x1 y1 z1
        ...
        typeN-1 xN-1 yN-1 zN-1

    The first line is the number of particles ``N``. The second line defines the
    orthorhombic simulation box with ``L = (Lx,Ly,Lz)`` as three lattice vectors
    and the current ``counter`` as the time associated with the state in the
    trajectory. The next ``N`` lines define the type and position of each particle.

    Parameters
    ----------
    filename : str
        Name of the file.
    mode : str
        Mode for opening ``filename`` (must be ``'w''`` or ``'a'``).

    .. note::

        The file handle will be kept open once the :class:`XYZWriter` is created.

    """
    def __init__(self, filename, mode='w'):
        self._handle = open(filename,mode)

    def write(self, state):
        """Write a frame to file.

        The new state will be written to ``filename`` and flushed to disk.

        .. note::

            Because the :class:`~learnmolsim.state.State` does not currently support a particle type,
            each particle is assigned a nominal type ``A``.

        Parameters
        ----------
        state : :class:`~learnmolsim.state.State`
            Simulation state.

        """
        self._handle.write('{:d}\n'.format(state.N))
        self._handle.write('Lattice="{:f} 0.0 0.0 0.0 {:f} 0.0 0.0 0.0 {:f}" Time={:d}\n'.format(state.box.L[0],state.box.L[1],state.box.L[2],state.counter))
        for r in state.positions:
            self._handle.write('A {:f} {:f} {:f}\n'.format(r[0],r[1],r[2]))
        self._handle.flush()
