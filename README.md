# PyMagneticFilament
This is a code for simulating an uniformly magnetized filament under rotating magnetic field.



## Validation & Convergence
* Filament implementation is validated against the Euler-Bernoulli beam solution, and script is given in 
[filament validation](./example/filament_validation.py). We have implemented Euler-Forward and Runge-Kutta 4 time-steppers. Although the RK4 converged to the analytical solution at the steady-state, we could not determine a stable time-step for Euler-forward.

![beam_deflection_stability](https://user-images.githubusercontent.com/53585636/167277472-2461d861-bc17-47da-9a81-78d87fa9107b.png)



* [Convergence study](./example/filament_convergence.py) is done by changing the number of elements of a filament and error is compared 
against the Euler-Bernolli beam solution.

![convergence_filament](https://user-images.githubusercontent.com/53585636/167277435-d1e9282c-df43-42d2-a7aa-83ad19fa9b9f.png)


## Magnetic filament simulations
Magnetic filament simulation scripts can be found [here](./example/magnetic_filament_simulation.py).
In this study we changed the magnetic field angular rotation speed. 



https://user-images.githubusercontent.com/53585636/167277341-153c86d2-f8de-4d42-b01e-d77a40b9378e.mp4

