import Bloch.BlochSolver as BlochSolver
import Mesh.YeeMesh as YeeMesh
import EMF.EMF as EMF
import Misc.Units as Units
import numpy as np
import time

if __name__ == '__main__':

    """
    Entry point for Maxwell-Bloch simulation. Define problem settings here
    and, if necessary, adjust units conversions.
    Due to numpy limitations, even for 1D simulations, the two remaining axis
    must have at least 2 points.
    """

    # Grid parameters and definition
    Nx = 1000                   # X-axis number of points
    Ny = 2                      # X-axis number of points
    Nz = 2                      # X-axis number of points
    dx = np.float64(1.0e-9)

    t_width = np.float64(1.0e-15)   # Total simulation temporal window
    dt = np.float64(1.0e-18)        # Integration step

    StorageSkip = 10    # Number of iterations between data storage

    # EMF parameters - Gaussian plane wave
    Direction = 0                       # Axis of EMF propagation
    EPol = 2                            # Axis of electric field polarization
    E0 = np.float64(1.0e11)             # Electric field amplitude
    EMFFreq = np.float64(0.6e15)        # EMF frequency
    Phase = np.float64(0.0)             # Initial phase
    FWHM = np.float64(Nx * dx / 10.0)   # Full Width Half Maximum
    R0 = np.float64(Nx * dx / 4.0)      # Displacement of the gaussian

    # Material parameters - Density operator properties
    levels = 2                              # Number of levels in the system
    eta = 1.0e21                            # Atomic density
    mu12 = 1.0e-29                          # Dipole moment
    omega12 = 2.0 * Units.PI * EMFFreq      # Transition frequency
    gamma12 = np.float64(1.0e15)            # Decay frequency

    mu = np.array([     [0.0, 0.0, -mu12, mu12],
                        [0.0, 0.0, mu12, -mu12],
                        [-mu12, mu12, 0.0, 0.0],
                        [mu12, -mu12, 0.0, 0.0]])           # Dipole transition matrix

    omega = np.array([  [0.0, 0.0, omega12, -omega12],
                        [0.0, 0.0, omega12, -omega12],
                        [-omega12, -omega12, 0.0, 0.0],
                        [omega12, omega12, 0.0, 0.0]])      # Level transition frequency matrix

    gamma = np.array([  [0.0, gamma12, 0.0, 0.0],
                        [0.0, -gamma12, 0.0, 0.0],
                        [0.0, 0.0, -gamma12 / 2.0, 0.0],
                        [0.0, 0.0, -gamma12 / 2.0, 0.0]])   # Decay matrix

    polTerms = np.array([0.0, 0.0, omega12, 0.0])   # Terms for polarization

    baseChangeFreq = np.array([0.0, 0.0, mu12, 0.0])    # Terms for Schrodinger-Picture

    initValues = np.array([0.0, 1.0, 0.0, 0.0])         # Initialization values for the density operator

    ########################### Conversion to natural units ###################################

    # Conversion of EMF constants
    E0 = np.float64(Units.convertToNatural(E0, ['M', 'L', 'T', 'C'], [1, 1, -2, -1]))
    EMFFreq = np.float64(Units.convertToNatural(EMFFreq, ['T'], [-1]))
    FWHM = np.float64(Units.convertToNatural(FWHM, ['L'], [1]))
    R0 = np.float64(Units.convertToNatural(R0, ['L'], [1]))

    # Conversion of material constants
    for i in range(mu.shape[0]):
        polTerms[i] = np.float64(Units.convertToNatural(polTerms[i], ['C', 'L'], [1, 1]))
        baseChangeFreq[i] = np.float64(Units.convertToNatural(baseChangeFreq[i], ['T'], [-1]))
        for j in range(mu.shape[1]):
            mu[i][j] = np.float64(Units.convertToNatural(mu[i][j], ['C', 'L'], [1, 1]))
            omega[i][j] = np.float64(Units.convertToNatural(omega[i][j], ['T'], [-1]))
            gamma[i][j] = np.float64(Units.convertToNatural(gamma[i][j], ['T'], [-1]))
    eta = np.float64(Units.convertToNatural(eta, ['L'], [-3]))

    ####################################### Simulation #####################################

    # Instantiate class objects
    Mesh = YeeMesh.YeeMesh(num_cells=[Nx, 2, 2], spacings=[dx] * 3, dt=dt)
    EMField = EMF.EMF(Mesh)
    DensOp = BlochSolver.BlochSolver(Mesh, levels, mu, omega, gamma, polTerms, baseChangeFreq, eta)

    # Initialize the EMField
    EMField.addGaussianPlaneWave(E0, EMFFreq, Phase, FWHM, R0)

    # Initialize DensOp
    DensOp.initialize(EMField.E, initValues)

    # Prepare auxiliary simulation variables
    maxIter = t_width / Mesh.dt + 1
    EFieldEvolution = []
    Rho11Evolution = []
    Rho22Evolution = []
    Rho12AbsEvolution = []
    Rho12PhaseEvolution = []

    # Run simulation
    i = 0
    start = time.time()
    while i < maxIter:
        EMField.push()
        DensOp.push(EMField.E, i * Mesh.dt)
        # Store data on selected iterations
        if i % StorageSkip == 0:
            print(i)
            EFieldEvolution.append(Mesh.face_fields[EMField.E][2, ...])
            Rho11Evolution.append(Mesh.center_tensor_fields[DensOp.RhoSchrodingerPicture][0, 0, ...])
            Rho22Evolution.append(Mesh.center_tensor_fields[DensOp.RhoSchrodingerPicture][1, 0, ...])
            Rho12AbsEvolution.append(np.abs(Mesh.center_tensor_fields[DensOp.RhoSchrodingerPicture][2, 0, ...]))
            Rho12PhaseEvolution.append(
                np.angle(Mesh.center_tensor_fields[DensOp.RhoSchrodingerPicture][2, 0, ...], deg=True))
        i += 1
    end = time.time()
    print("Simulation took " + str(end - start) + " seconds")

