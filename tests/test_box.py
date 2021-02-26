import numpy as np
import pytest
import learnmolsim as lms

@pytest.fixture
def ortho():
    return lms.state.Box([10.0,15.0,20.0])

def test_init(ortho):
    assert np.allclose(ortho.L, [10.,15.,20.])

def test_init_cube():
    c = lms.state.Box(10.)
    assert np.allclose(c.L, [10.,10.,10.])

def test_init_wrongsize():
    with pytest.raises(TypeError):
        lms.state.Box((10.,15.))

def test_set_wrongsize(ortho):
    with pytest.raises(TypeError):
        ortho.L = [10.,15.]

def test_init_wrongval():
    with pytest.raises(ValueError):
        lms.state.Box([1.0,-2.0,3.0])

def test_set_wrongval(ortho):
    with pytest.raises(ValueError):
        ortho.L = [1.0,-2.0,3.0]

def test_wrap(ortho):
    # wrap in place
    x = [11,-1,18]
    im = [0,0,0]
    x,im = ortho.wrap(x,im)
    assert np.allclose(x, [1,14,18])
    assert np.allclose(im, [1,-1,0])

def test_wrap_multiple_particles(ortho):
    # multiple particles
    x = [[1,17,3],[15,2,-2]]
    im = [[1,2,3],[4,5,6]]
    x,im = ortho.wrap(x,im)
    assert np.allclose(x, [[1,2,3],[5,2,18]])
    assert np.allclose(im, [[1,3,3],[5,5,5]])

def test_wrap_multiple_images(ortho):
    # wrap multiple images
    x = [30,30,30]
    im = [1,2,3]
    x,im = ortho.wrap(x,im)
    assert np.allclose(x, [0,0,10])
    assert np.allclose(im, [4,4,4])

def test_wrap_noimage(ortho):
    # ignore image argument
    x = [1,2,3]
    x = ortho.wrap(x)
    assert np.allclose(x, [1,2,3])

def test_wrap_wrongshape(ortho):
    # just positions
    with pytest.raises(TypeError):
        ortho.wrap([1,2])
    with pytest.raises(TypeError):
        ortho.wrap([1,2,3,4])
    with pytest.raises(TypeError):
        ortho.wrap([[1,2],[3,4]])

    # with images
    with pytest.raises(TypeError):
        ortho.wrap([1,2,3],[0,0])
    with pytest.raises(TypeError):
        ortho.wrap([1,2,3],[0,0,0,0])
    with pytest.raises(TypeError):
        ortho.wrap([[1,2,3],[4,5,6]],[[1,2],[3,4]])

def test_minimum_image(ortho):
    x0 = np.array([1,14,2])
    x1 = np.array([8,1,17])
    dx = x1-x0
    dx = ortho.minimum_image(dx)
    assert np.allclose(dx, [-3,2,-5])

def test_minimum_image_multiple_particles(ortho):
    x0 = np.array([1,14,2])
    x1 = np.array([8,1,17])
    x2 = x0 + [1,-1,1]

    dx = np.array([x1-x0,x2-x0])
    dx = ortho.minimum_image(dx)
    assert np.allclose(dx, [[-3,2,-5],[1,-1,1]])

def test_minimum_image_multiple_images(ortho):
    dx = ortho.minimum_image([-14.0,30.0,45.0])
    assert np.allclose(dx, [-4.0,0.0,5.0])
