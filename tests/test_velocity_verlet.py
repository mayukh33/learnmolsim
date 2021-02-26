import numpy as np
import pytest
import learnmolsim as lms

class ConstantPotential:
    """Dummy potential for testing integrator.

    Parameters
    ----------
    f : array_like
        Force to apply to each particle.

    """
    def __init__(self, f):
        self.f = np.array(f,dtype=np.float64)

    def compute(self, state):
        u = -np.sum(self.f*state.positions,axis=1)
        f = self.f*np.ones((state.N,3))

        return u,f

@pytest.fixture
def ig():
    return lms.dynamics.VelocityVerlet(0.1,ConstantPotential((0,0,0)))

def test_init(ig):
    assert ig.dt == pytest.approx(0.1)
    assert isinstance(ig.potential,ConstantPotential)

    ig.dt = 0.2
    assert ig.dt == pytest.approx(0.2)

    with pytest.raises(ValueError):
        ig.dt = -0.1

def test_advance(ig):
    s = lms.state.State(1,lms.state.Box(10.0),mass=10.0)
    s.positions = [[0,9.9,5]]
    s.velocities = [[-1,2,1]]

    # advance with no force
    ig.advance(s)
    assert np.allclose(s.positions,[[9.9,0.1,5.1]])
    assert np.allclose(s.velocities,[[-1,2,1]])

    # add a constant force to the potential, reset the system state
    ig.potential.f = (10,-20,30)
    s.positions = [[0,0,0]]
    s.velocities = [[0,0,0]]
    s.forces = None
    ig.advance(s)

    # dt*f/m
    assert np.allclose(s.velocities, [[0.1,-0.2,0.3]])
    # half that times dt again because only use the half-step velocity
    assert np.allclose(s.positions, [[0.005,9.99,0.015]])
