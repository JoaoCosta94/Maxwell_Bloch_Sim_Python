# Simulator of the Maxwell-Bloch equations

This code was developed as a prototype for a solver of the Maxwell-Bloch equations to test the viability of
the numerical methods and the software architecture. An High-Performance version of this code using C++ and CUDA
is currently under development.

# Methods

The solver uses an Yee-Mesh where all the information about the spatial components of the physical quantities
involved is stored. The Bloch Solver uses a RK2 scheme while the Maxwell Solver uses a leapfrog scheme.
The solver was designed to be modular and to be possible integrated with other solvers such as CFD or PIC.
