import numpy as np
from tqdm import tqdm


def integrate(system, time_stepper, total_steps, dt):

    time = 0.0
    step = 0
    # First call back is to save initial state
    system.callback(time, step)

    for step in tqdm(range(1, total_steps + 1)):

        # Apply torques
        system.apply_forces_or_torques(time)

        # Time integration
        time_stepper(system, time, dt)

        # Update time
        time += dt

        # Call back
        system.callback(time, step)

        if np.isnan(system.state).any() == True:
            print("Nan detected time " + str(time))
            return
