import numpy as np
import Misc.Units as Units

class EMF():
    def __init__(self, Mesh):
        """
        EMF is an object that solves the Maxwell equations using an
        YeeMesh scheme. The Electric field is represented on the cell
        faces and the Magnetic field is represented on the cell edges

        :param Mesh:        Simulations YeeMesg
        """
        self.Mesh = Mesh

        self.E = self.Mesh.add_face_field(dtype=np.float64)
        self.B = self.Mesh.add_edge_field(dtype=np.float64)

    def addGaussianPlaneWave(self, E0, Freq, Phase, FWHM, R0, Direction = 0, EPol = 2):
        """
        Function that initializes the electro-magnetic field with
        a gaussian profile

        :param E0:          Electric field amplitude
        :param Freq:        Wave frequency
        :param Phase:       Initial phase
        :param FWHM:        Full Width Half Maximum
        :param R0:          Initial displacement
        """
        # Hardcoded for propagation along the positive x axis
        self.Mesh.face_fields[self.E][2] = (E0 * np.sin(2.0*Units.PI*Freq * self.Mesh.faceZ[0] + Phase)
                                            * np.exp(np.power(self.Mesh.faceZ[0] - R0, 2) / 2.0 * FWHM ** 2))
        self.Mesh.edge_fields[self.B][1] = (E0 / Units.C * np.sin(2.0*Units.PI*Freq * self.Mesh.edgeY[0] + Phase)
                                            * np.exp(np.power(self.Mesh.edgeY[0] - R0, 2) / 2.0 * FWHM ** 2))
        self.initialize()

    def initialize(self):
        """
        Function to be called after the creation of a Electro-Magnetic field.
        Puts the magnetic field on Leap-Frog with the electric field
        """
        # Evolving B to t = 1/2
        self.Mesh.curl_face_to_edge(self.E, self.B, '+=', -0.5 * self.Mesh.dt)

    def push(self):
        """
        Evolves the magnetic and electrical fields by YeeMesh.dt
        """
        self.Mesh.curl_edge_to_face(self.B, self.E, '+=', self.Mesh.dt * Units.C ** 2)
        self.Mesh.curl_face_to_edge(self.E, self.B, '+=', -self.Mesh.dt)


# if __name__ == '__main__':
#     """
#     Main function for simple testing
#     """
#     x_width = 5.0e-6
#     dx = np.float64(1.0e-8)
#     t_width = np.float64(1.0e-15)
#     dt = np.float64(1.0e-18)
#
#     constants = {
#         "Ep0": np.float64(1.0e11),
#         "v": np.float64(3.0e8),
#         "lambdaField": np.float64(500.0e-9),
#         "probeWidth": np.float64(x_width / 10.0),
#         "probeDisplacement": np.float64(x_width / 3.0),
#         "dt": dt,
#         "scaling": None
#     }
#
#     ###################################################################################################################
#
#     Ct = np.float64(1.0 / 6.58e-16)
#     Cl = np.float64(1.0 / 1.97e-7)
#     Cc = np.float64(1.0 / 1.88e-18)
#     Ck = np.float64(1.0 / 1.78e-36)
#
#     x_width *= Cl
#     dx *= Cl
#     t_width *= Ct
#     dt *= Ct
#     snapMultiple = 5
#
#     constants["Ep0"] *= Ck * Cl / (Ct ** 2 * Cc)
#     constants["v"] = 1.0
#     constants["lambdaField"] *= Cl
#     constants["probeWidth"] *= Cl
#     constants["probeDisplacement"] *= Cl
#     constants["dt"] *= Ct
#     constants["scaling"] = Ck * Cl / (Ct ** 2 * Cc)
#
#     ####################################################################################################################
#
#     X = np.arange(0.0, x_width + dx, dx)
#     T = np.arange(0.0, t_width + dt, dt)
#
#     mesh = YeeMesh(num_cells=[len(X), 2, 2], spacings=[dx] * 3)
#     field = EMF(constants)
#     field.initialize_gaussian_field(mesh)
#
#     CENTER_FIELD_X = mesh.add_center_field()
#     CENTER_FIELD_Y = mesh.add_center_field()
#     CENTER_FIELD_Z = mesh.add_center_field()
#     mesh.interpolate_face_to_center(field.E, CENTER_FIELD_X, CENTER_FIELD_Y, CENTER_FIELD_Z)
#
#     FIELD_EVOL = np.zeros((len(T), len(X)), dtype=np.float64)
#     FIELD_EVOL[0] = np.abs(mesh.center_fields[CENTER_FIELD_Z][:, 0, 0] / field.scaling)
#
#     for i in range(1, len(T)):
#         field.push()
#         mesh.interpolate_face_to_center(field.E, CENTER_FIELD_X, CENTER_FIELD_Y, CENTER_FIELD_Z)
#         FIELD_EVOL[i] = np.abs(mesh.center_fields[CENTER_FIELD_Z][:, 0, 0] / field.scaling)
#         print("{:.3f}".format(T[i] / t_width * 100) + '%')
#
#     X, T = np.meshgrid(X/Cl, T/Ct)
#     scale = np.linspace(np.min(FIELD_EVOL), np.max(FIELD_EVOL), 100)
#
#     pyplot.figure()
#     pyplot.title("EMF_EVOL")
#     pyplot.xlabel("X")
#     pyplot.ylabel("T")
#     pyplot.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
#     pyplot.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#     pyplot.contourf(X, T, FIELD_EVOL, levels=scale, cmap=pyplot.get_cmap("inferno"))
#     pyplot.colorbar()
#     pyplot.show()
