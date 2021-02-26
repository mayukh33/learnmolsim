import numpy as np
import pytest
import learnmolsim as lms

@pytest.fixture
def state():
    return lms.state.State(2,lms.state.Box(10.0))

def test_init_default(state):
    assert state.N == 2
    assert state.mass == pytest.approx(1.0)
    assert state.counter == 0

def test_init_nodefault():
    box = lms.state.Box(3.0)
    state = lms.state.State(3,box,mass=3.0,counter=1)
    assert state.N == 3
    assert state.box is box
    assert state.mass == pytest.approx(3.0)
    assert state.counter == 1

def test_box(state):
    assert isinstance(state.box,lms.state.Box)
    assert np.allclose(state.box.L, [10,10,10])

    new_box = lms.state.Box([4,5,6])
    state.box = new_box
    assert state.box is new_box
    assert np.allclose(state.box.L, [4,5,6])

def test_mass(state):
    state.mass = 2.0
    assert state.mass == pytest.approx(2.0)

def test_counter(state):
    state.counter += 1
    assert state.counter == 1

def test_positions(state):
    assert np.allclose(state.positions,[[0,0,0],[0,0,0]])

    state.positions = np.ones((state.N,3))
    assert np.allclose(state.positions, [[1,1,1],[1,1,1]])

    state.positions[0] = [2,3,4]
    assert np.allclose(state.positions, [[2,3,4],[1,1,1]])

    with pytest.raises(TypeError):
        state.positions = [1,2,3]

def test_images(state):
    assert np.allclose(state.images,[[0,0,0],[0,0,0]])

    state.images = np.ones((state.N,3))
    assert np.allclose(state.images, [[1,1,1],[1,1,1]])

    state.images[0] = [2,3,4]
    assert np.allclose(state.images, [[2,3,4],[1,1,1]])

    with pytest.raises(TypeError):
        state.images = [1,2,3]

def test_velocities(state):
    assert state.velocities is None

    state.velocities = np.zeros((state.N,3))
    assert np.allclose(state.velocities,[[0,0,0],[0,0,0]])

    state.velocities[0] = [2,3,4]
    assert np.allclose(state.velocities, [[2,3,4],[0,0,0]])

    state.velocities = None

    with pytest.raises(TypeError):
        state.velocities = [1,2,3]

def test_energies(state):
    assert state.energies is None

    state.energies = np.zeros(state.N)
    assert np.allclose(state.energies,[0,0])

    state.energies[0] = 1.
    assert np.allclose(state.energies, [1,0])

    state.energies = None

    with pytest.raises(TypeError):
        state.energies = [[1,2],[3,4]]

def test_forces(state):
    assert state.forces is None

    state.forces = np.zeros((state.N,3))
    assert np.allclose(state.forces,[[0,0,0],[0,0,0]])

    state.forces[0] = [2,3,4]
    assert np.allclose(state.forces, [[2,3,4],[0,0,0]])

    state.forces = None

    with pytest.raises(TypeError):
        state.forces = [1,2,3]
