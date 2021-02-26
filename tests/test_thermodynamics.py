import numpy as np
import pytest
import learnmolsim as lms

@pytest.fixture
def state():
    s = lms.state.State(2,lms.state.Box(10.0),mass=10.0)
    s.positions = [[1,1,1],[2,2,2]]
    s.velocities = [[1,-1,1],[-2,2,-2]]
    s.energies = [3,-4]
    s.forces = [[1,2,3],[-1,-2,-3]]
    return s

@pytest.fixture
def thermo():
    return lms.analyze.Thermodynamics()

def test_kinetic_energy(state,thermo):
    assert thermo.kinetic_energy(state) == pytest.approx(75.)

def test_potential_energy(state,thermo):
    assert thermo.potential_energy(state) == pytest.approx(-1.)

def test_kT(state,thermo):
    assert thermo.kT(state) == pytest.approx(25.)

def test_pressure(state,thermo):
    # ideal gas + virial
    pid = 2*25/10**3
    pex = (6-12)/(3*10**3)

    assert thermo.pressure(state) == pytest.approx(pid+pex)
