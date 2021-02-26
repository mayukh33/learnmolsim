"""Example molecular dynamics simulation of nearly hard spheres."""
import sys
sys.path.append('..')
import numpy as np
import learnmolsim as lms

# create empty cubic box
kT_target = 1.5
N = 50
box = lms.state.Box(10.0)
state = lms.state.State(N,box)

# insert particles randomly without overlap
for i in range(state.N):
    found = False
    attempt = 0
    max_attempts = 100
    while not found and attempt < max_attempts:
        ri = state.box.L*np.random.uniform(0.0,1.0,3)
        dr = state.box.minimum_image(state.positions[:i] - ri)

        found = (i == 0 or np.all(np.linalg.norm(dr,axis=1) >= 1.0))

        if found:
            state.positions[i] = ri
        attempt += 1

    if not found:
        raise RuntimeError('Unable to place all particles')

# randomize the velocities with zero initial mean and temperature kT_target
state.velocities = np.random.normal(0.0,np.sqrt(kT_target/state.mass),(state.N,3))
state.velocities -= np.mean(state.velocities, axis=0)

# WCA potential + Verlet integration
lj = lms.potential.LennardJones(1.0,1.0,2**(1./6.),shift=True)
nve = lms.dynamics.VelocityVerlet(0.005,lj)

# analysis tools
analyze_every = 50
thermo = lms.analyze.Thermodynamics()
xyz = lms.write.XYZWriter('lj.xyz')

# equilibration with aggressive isokinetic thermostat
with open('thermo.log','w') as f:
    # ensure system forces are initialized before starting run, or analysis will fail
    state.energies,state.forces = lj.compute(state)

    # advance 5000 steps
    while state.counter < 5000:
        # perform analysis at the beginning of the step
        if state.counter % analyze_every == 0:
            # report the counter
            print(state.counter)

            # dump state
            xyz.write(state)

            # compute thermo properties
            kT = thermo.kT(state)
            P = thermo.pressure(state)
            f.write('{} {} {}\n'.format(state.counter,kT,P))
            f.flush()

        # integration step
        nve.advance(state)

        # rescale the temperature if it wanders too far (bad thermostat, but it works OK here)
        kT = thermo.kT(state)
        if np.abs(kT-kT_target)/kT_target > 0.02:
            state.velocities *= np.sqrt(kT_target/kT)

