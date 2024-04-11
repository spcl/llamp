### Running ICON Experiments

#### Quick start
Clone the ICON repository and enter the following commands:
```console
> cp icon-src/mo_mpi.f90 <icon-root>/src/parallel_infrastructure/
> cp icon-src/mo_atmo_nonhydrostatic.f90 <icon-root>/src/drivers/
```
After compiling ICON, you can run it for one trial with the corresponding run script. Remember to change the paths in the script as per ICON's installation path.
