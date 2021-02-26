import numpy as np
import pytest
import learnmolsim as lms

@pytest.fixture
def lj():
    return lms.potential.LennardJones(1.5,0.5,2.5)

def test_init(lj):
    assert lj.epsilon == pytest.approx(1.5)
    assert lj.sigma == pytest.approx(0.5)
    assert lj.rcut == pytest.approx(2.5)

def test_compute(lj):
    """Test Lennard-Jones energy."""

    b = lms.state.Box(10)
    s = lms.state.State(3,b)
    s.positions = [[0,0,0],[0,0.5,0],[0,0,9.5]]

    u,f = lj.compute(s)

    assert np.sum(u) == pytest.approx(4*1.5*(2**-6-2**-3))
    assert np.sum(f,axis=0) == pytest.approx(0)

def test_energy_force(lj):
    u,f = lj.energy_force(0.5**2)
    assert u == pytest.approx(0.0)
    assert f == pytest.approx(24*1.5/0.5**2)

    r = np.array([0.5*2**(1./6.),3.0,0.0])
    u,f = lj.energy_force(r**2)

    assert np.allclose(u,[-1.5,0,np.inf])
    assert np.allclose(f,[0,0,np.inf])

def test_energy_shift(lj):
    rmin = 0.5*2.**(1./6.)
    lj.rcut = rmin
    lj.shift = False

    rsq = [rmin**2,1.5**2]
    u,_ = lj.energy_force(rsq)
    assert np.allclose(u, [-1.5,0])

    lj.shift = True
    u,_ = lj.energy_force(rsq)
    assert np.allclose(u, [0,0])
