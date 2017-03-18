import numpy as np
import Misc.Units as Units

class BlochSolver():
    def __init__(self, Mesh, Levels, Mu, Omega, Gamma, PolTerms, BaseChangeFreq, Eta):
        """
        BlochSolver is an object that contains attributes and methods to study
        the dynamics of a system described by Bloch equations. This solver
        focuses specifically on the evolution of the quantum density operator
        which is described by Bloch equations. This solver does not use the R.W.A.
        The variables of interest are RhoSchrodingerPicture and Polarization, which
        point the respective center tensors on a YeeMesh object

        :param Mesh:                YeeMesh object used for the simulation
        :param Levels:              Number of atomic levels in the system
        :param Mu:                  Dipole moments matrix
        :param Omega:               Transition frequencies matrix
        :param Gamma:               Spontaneous decay frequencies matrix
        :param PolTerms:            Terms for calculating the polarization with the density operator
        :param BaseChangeFreq:      Frequencies for conversion from interaction picture to schrodinger's picture
        :param Eta:                 Atomic density
        """
        self.Mesh = Mesh
        self.Levels = Levels
        self.CouplingMatrix = Mu / Units.HBAR
        self.Omega = Omega
        self.Gamma = Gamma
        self.PolTerms = PolTerms
        self.BaseChangeFreq = BaseChangeFreq
        self.Eta = Eta

        self.Dim = self.Levels ** 2
        self.RhoInteractionPicture = None
        self.RhoSchrodingerPicture = None
        self.CenterEField = None
        self.PolarizationAmplitude = None
        self.Polarization = None
        self.DynamicMatrix = None

        if ( Mu.shape[0] != self.Dim or Mu.shape[1] != self.Dim or
             Omega.shape[0] != self.Dim or Omega.shape[1] != self.Dim or
             Gamma.shape[0] != self.Dim or Gamma.shape[1] != self.Dim or
             PolTerms.shape[0] != self.Dim or BaseChangeFreq.shape[0] != self.Dim):
            raise Exception("BlochSolver.BlochSolver - Invalid dimensions for input arguments")

    def schrodingerPictue(self, t):
        """
        Function that converts the system's state from the interaction picture
        to the Schrodinger picture
        :param t:   Temporal instant
        """
        for i in range(self.Dim):
            self.Mesh.center_tensor_fields[self.RhoSchrodingerPicture][i, 0, ...] = (self.Mesh.center_tensor_fields[self.RhoSchrodingerPicture][i, 0, ...]
                                                                                     * np.exp(1.0j * self.BaseChangeFreq[i] * t))

    def calculatePolarization(self, EField):
        """
        Function that updates the current state of the polarization
        :param EField:      Center vector field on the Mesh to be used for the calculation
        """
        # Calculation of the Polarization amplitude
        self.PolarizationAmplitude = np.zeros((self.Mesh.num_cells[0], self.Mesh.num_cells[1], self.Mesh.num_cells[2]),
                                              dtype=np.float64)
        for i in range(self.Dim):
            if self.PolTerms[i] != 0:
                self.PolarizationAmplitude += self.Eta * self.PolTerms[i] * 2.0 * np.real(self.Mesh.center_tensor_fields[self.RhoSchrodingerPicture][i, 0, ...])

        # Calculation of the Field Magnitude
        EPS = 1.0e-15 # Margin for div by zero
        Aux = self.Mesh.center_vector_fields[EField] * self.Mesh.center_vector_fields[EField]
        Magnitude = np.sqrt(Aux[0] + Aux[1] + Aux[2]) + EPS

        # Update the polarization
        for i in range(3):
            self.Mesh.center_vector_fields[self.Polarization][i, ...] = (self.PolarizationAmplitude *
                                                                         self.Mesh.center_vector_fields[EField][i, ...]
                                                                         / Magnitude)

    def initialize(self, EField, InitValues):
        """
        Function that initializes the density operator
        :param EField:          Field for calculating the initial polarization
        :param InitValues:      Initial values for the each density operator component
        """
        if InitValues.shape[0] != self.Dim:
            raise Exception("BlochSolver.Initialize - Invalid dimension for initialization values")

        # Initialize the density operator tensor field
        self.RhoSchrodingerPicture = self.Mesh.add_center_tensor_field((self.Dim, 1), dtype=np.complex128)
        self.RhoInteractionPicture = self.Mesh.add_center_tensor_field((self.Dim, 1), dtype=np.complex128)
        for i in range(self.Dim):
            self.Mesh.center_tensor_fields[self.RhoSchrodingerPicture][i, 0, ...] = InitValues[i]
            self.Mesh.center_tensor_fields[self.RhoInteractionPicture][i, 0, ...] = InitValues[i]

        # Initialize aux fields
        self.CenterEField = self.Mesh.add_center_vector_field(dtype=np.float64)
        self.Polarization = self.Mesh.add_center_vector_field(dtype=np.float64)
        self.DynamicMatrix = self.Mesh.add_center_tensor_field((self.Dim, self.Dim), dtype=np.complex128)

        # Update initial polarization
        self.Mesh.interpolate_face_to_center_vector(EField, self.CenterEField)
        self.calculatePolarization(self.CenterEField)

    def updateDynamicMatrix(self, EField, t):
        """
        Function that updates the dynamic matrix
        in the equation system
        :param EField:  Center vector field on the Mesh to be used for evolving the system
        :param t:       Temporal instant
        """
        for i in range(self.Dim):
            for j in range(self.Dim):
                if self.CouplingMatrix[i][j] != 0.0:
                    Aux = self.Mesh.center_vector_fields[EField] * self.Mesh.center_vector_fields[EField]
                    Aux = 1.0j * self.CouplingMatrix[i][j] * np.sqrt(Aux[0] + Aux[1] + Aux[2])
                    self.Mesh.center_tensor_fields[self.DynamicMatrix][i, j, ...] = Aux * np.exp(1.0j * self.Omega[i][j] * t)

    def push(self, EField, t):
        """
        Function that evolves the system's state using a Runke-Kutta 2 method
        :param EField:      Face field on the Mesh to be used for evolving the system
        :param t:           Current temporal instant
        """
        # Interpolate EField to center of the cell
        self.Mesh.interpolate_face_to_center_vector(EField, self.CenterEField)

        # First RK2 step
        self.updateDynamicMatrix(self.CenterEField, t)

        self.Mesh.mul_center_tensor_fields(self.DynamicMatrix, self.RhoInteractionPicture, self.RhoSchrodingerPicture,
                                           mode="=", scalar=0.5 * self.Mesh.dt)
        self.Mesh.mul_external_matrix_by_center_tensor(self.Gamma, self.RhoInteractionPicture, self.RhoSchrodingerPicture,
                                                       mode='+=', scalar=0.5 * self.Mesh.dt)
        self.Mesh.add_over_center_tensors(self.RhoInteractionPicture, self.RhoSchrodingerPicture, mode='+=')

        # Second Rk2 step
        self.updateDynamicMatrix(self.CenterEField, t + 0.5 * self.Mesh.dt)

        self.Mesh.mul_center_tensor_fields(self.DynamicMatrix, self.RhoSchrodingerPicture, self.RhoInteractionPicture,
                                           mode="+=", scalar=self.Mesh.dt)
        self.Mesh.mul_external_matrix_by_center_tensor(self.Gamma, self.RhoSchrodingerPicture, self.RhoInteractionPicture,
                                                       mode='+=', scalar=self.Mesh.dt)

        # Update Schrodinger Picture
        self.schrodingerPictue(t+self.Mesh.dt)

        # Update Polarization
        self.calculatePolarization(self.CenterEField)
