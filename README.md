# PyMagneticFilament
This is a code for simulating an uniformly magnetized filament under rotating magnetic field.



## Validation & Convergence
* Filament implementation is validated against the Euler-Bernoulli beam solution, and script is given in 
[filament validation](./example/filament_validation.py)

* [Convergence study](./example/filament_convergence.py) is done by changing the number of elements of a filament and error is compared 
against the Euler-Bernolli beam solution.

## Magnetic filament simulations
Magnetic filament simulation scripts can be found [here](./example/magnetic_filament_simulation.py).
In this study we changed the magnetic field angular rotation speed. 

