import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import time
import warnings


def idx(i, j, k, Li, Lj, Lk):
    i = i % Li
    j = j % Lj
    k = k % Lk
    return (i*Lj + j)*Lk + k


def multi_roll(array, rolls):
    if len(rolls) < 1:
        raise ValueError("The rolling values can not be empty.")
    if len(array.shape) != len(rolls):
        raise ValueError("Rolling values %s don't match the shape of the array %s" % (str(rolls), str(array.shape)))

    ret = np.roll(array, rolls[0], 0)
    for i in range(1, len(rolls)):
        ret = np.roll(ret, rolls[i], i)
    return ret


class Center_Function:
    def __init__(self, function, position, time_ref):
        self.function = function
        self.pos = position
        self.time_ref = time_ref

    def __add__(self, other):
        return other + self.function(self.pos[0], self.pos[1], self.pos[2], self.time_ref[0])


class Face_Function:
    def __init__(self, functionX, functionY, functionZ, positionX, positionY, positionZ, time_ref):
        self.functionX = functionX
        self.functionY = functionY
        self.functionZ = functionZ
        self.posX = positionX
        self.posY = positionY
        self.posZ = positionZ
        self.time_ref = time_ref

    def __add__(self, other):
        ret = np.zeros_like(other)
        ret += other
        ret[0] += self.functionX(self.posX[0], self.posX[1], self.posX[2], self.time_ref[0])
        ret[1] += self.functionY(self.posY[0], self.posY[1], self.posY[2], self.time_ref[0])
        ret[2] += self.functionZ(self.posZ[0], self.posZ[1], self.posZ[2], self.time_ref[0])
        return ret

class YeeMesh:
    """
    Implements a three dimensional regular rectangular mesh and several numerical methods. The following fields can be sampled within the class:
        - center field: scalar field sampled on the geometric center of the each cell;
        - face field: vector field with each component sampled in the geometric center of the minimal faces perpendicular to them;
        - edge field: vector field with each component sampled in the geometric center of the minimal edges collinear with them;
        - center tensor field: general tensor with all components sampled at the geometric center of the each cell;
        - face tensor field: 3x3 tensor (matrix) sampled in the faces so as to directly multiply with the face vector fields;
        - edge tensor field: 3x3 tensor (matrix) sampled in the edges so as to directly multiply with the edge vector fields;
    """
    def __init__(self,
                 num_cells,
                 spacings,
                 dt,
                 center_fields_names = None,
                 edge_fields_names = None,
                 face_fields_names = None,
                 center_dtypes = np.float64,
                 edge_dtypes = np.float64,
                 face_dtypes = np.float64):

        self.num_cells = num_cells
        self.spacings = spacings
        self.dt = dt
        self.box_dims = (spacings[0]*num_cells[0], spacings[1]*num_cells[1], spacings[2]*num_cells[2])
        self.time = [0.0] # must be a list
        self.center = None
        self.edgeX = None
        self.edgeY = None
        self.edgeZ = None
        self.faceX = None
        self.faceY = None
        self.faceZ = None
        self.center_fields = {}
        self.center_vector_fields = {}
        self.center_tensor_fields = {}
        self.edge_fields = {}
        self.edge_tensor_fields = {}
        self.face_fields = {}
        self.face_tensor_fields = {}

        self.generate_grid(num_cells, spacings)
        self.allocate_fields(center_fields_names, edge_fields_names, face_fields_names, center_dtypes, edge_dtypes, face_dtypes)

    def generate_grid(self, num_cells, spacings):
        self.center = np.array(np.meshgrid(np.array([0.5 * spacings[0] + i * spacings[0] for i in range(num_cells[0])]),
                                           np.array([0.5 * spacings[1] + i * spacings[1] for i in range(num_cells[1])]),
                                           np.array([0.5 * spacings[2] + i * spacings[2] for i in range(num_cells[2])]),
                                           indexing='ij', copy=False))

        self.edgeX = np.array(np.meshgrid(np.array([0.5 * spacings[0] + i * spacings[0] for i in range(num_cells[0])]),
                                          np.array([i * spacings[1] for i in range(num_cells[1])]),
                                          np.array([i * spacings[2] for i in range(num_cells[2])]),
                                          indexing='ij', copy=False))

        self.edgeY = np.array(np.meshgrid(np.array([i * spacings[0] for i in range(num_cells[0])]),
                                          np.array([0.5 * spacings[1] + i * spacings[1] for i in range(num_cells[1])]),
                                          np.array([i * spacings[2] for i in range(num_cells[2])]),
                                          indexing='ij', copy=False))

        self.edgeZ = np.array(np.meshgrid(np.array([i * spacings[0] for i in range(num_cells[0])]),
                                          np.array([i * spacings[1] for i in range(num_cells[1])]),
                                          np.array([0.5 * spacings[2] + i * spacings[2] for i in range(num_cells[2])]),
                                          indexing='ij', copy=False))

        self.faceZ = np.array(np.meshgrid(np.array([0.5 * spacings[0] + i * spacings[0] for i in range(num_cells[0])]),
                                          np.array([0.5 * spacings[1] + i * spacings[1] for i in range(num_cells[1])]),
                                          np.array([i * spacings[2] for i in range(num_cells[2])]),
                                          indexing='ij', copy=False))

        self.faceX = np.array(np.meshgrid(np.array([i * spacings[0] for i in range(num_cells[0])]),
                                          np.array([0.5 * spacings[1] + i * spacings[1] for i in range(num_cells[1])]),
                                          np.array([0.5 * spacings[2] + i * spacings[2] for i in range(num_cells[2])]),
                                          indexing='ij', copy=False))

        self.faceY = np.array(np.meshgrid(np.array([0.5 * spacings[0] + i * spacings[0] for i in range(num_cells[0])]),
                                          np.array([i * spacings[1] for i in range(num_cells[1])]),
                                          np.array([0.5 * spacings[2] + i * spacings[2] for i in range(num_cells[2])]),
                                          indexing='ij', copy=False))

    def name_checker_generator(self, field_name, field_existing_names):
        if field_name is None:
            field_name = int(np.random.randint(2 ** 32, dtype=np.uint32))
            while True:
                if field_name not in field_existing_names:
                    break
                field_name = int(np.random.randint(2 ** 32, dtype=np.uint32))

        if field_name in field_existing_names:
            raise ValueError("Field name '%s' already in use." % (field_name,))

        return field_name

    def add_center_field(self, field_name = None, dtype = np.float64):
        field_name = self.name_checker_generator(field_name, self.center_fields.keys())
        self.center_fields[field_name] = np.zeros(self.num_cells, dtype)
        return field_name

    def add_center_vector_field(self, field_name = None, dtype = np.float64):
        field_name = self.name_checker_generator(field_name, self.center_vector_fields.keys())
        self.center_vector_fields[field_name] = np.zeros([3] + self.num_cells, dtype)
        return field_name


    def add_center_field_function(self, function, field_name = None, dtype = np.float64):
        field_name = self.name_checker_generator(field_name, self.center_fields.keys())
        self.center_fields[field_name] = Center_Function(function, self.center, self.time)
        return field_name

    def add_center_tensor_field(self, shape, field_name = None, dtype = np.float64):
        field_name = self.name_checker_generator(field_name, self.center_tensor_fields.keys())
        self.center_tensor_fields[field_name] = np.zeros(list(shape) + list(self.num_cells), dtype)
        return field_name

    def add_edge_field(self, field_name = None, dtype = np.float64):
        field_name = self.name_checker_generator(field_name, self.edge_fields.keys())
        self.edge_fields[field_name] = np.zeros([3] + list(self.num_cells), dtype)
        return field_name

    def add_face_field(self, field_name = None, dtype = np.float64):
        field_name = self.name_checker_generator(field_name, self.face_fields.keys())
        self.face_fields[field_name] = np.zeros([3] + list(self.num_cells), dtype)
        return field_name

    def allocate_fields(self, center_field_names, edge_field_names, face_field_names, center_dtypes, edge_dtypes, face_dtypes):
        if not hasattr(center_field_names, "__contains__"):
            if center_field_names is None:
                center_field_names = []
            elif isinstance(center_field_names, (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                warnings.warn("This code has been refactored for usage of strings with dictionaries to index the various fields. Initialization should now be done with a string or list of strings. However, initialization with an integer retains the same behaviour for backwards compatibility", Warning, stacklevel=3)
                if center_field_names == 0:
                    center_field_names = []
                else:
                    center_field_names = list(range(center_field_names))
            else:
                center_field_names = [center_field_names]

        if not hasattr(edge_field_names, "__contains__"):
            if center_field_names is None:
                center_field_names = []
            elif isinstance(edge_field_names, (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                warnings.warn("This code has been refactored for usage of strings with dictionaries to index the various fields. Initialization should now be done with a string or list of strings. However, initialization with an integer retains the same behaviour for backwards compatibility", Warning, stacklevel=3)
                if edge_field_names == 0:
                    edge_field_names = []
                else:
                    edge_field_names = list(range(edge_field_names))
            else:
                edge_field_names = [edge_field_names]

        if not hasattr(face_field_names, "__contains__"):
            if center_field_names is None:
                center_field_names = []
            elif isinstance(face_field_names, (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                warnings.warn("This code has been refactored for usage of strings with dictionaries to index the various fields. Initialization should now be done with a string or list of strings. However, initialization with an integer retains the same behaviour for backwards compatibility", Warning, stacklevel=3)
                if face_field_names == 0:
                    face_field_names = []
                else:
                    face_field_names = list(range(face_field_names))
            else:
                face_field_names = [face_field_names]

        if not hasattr(center_dtypes, "__contains__"):
            center_dtypes = [center_dtypes]*len(center_field_names)
        if not hasattr(edge_dtypes, "__contains__"):
            edge_dtypes = [edge_dtypes]*len(edge_field_names)
        if not hasattr(face_dtypes, "__contains__"):
            face_dtypes = [face_dtypes]*len(face_field_names)

        for i in range(len(center_field_names)):
            self.add_center_field(center_field_names[i], center_dtypes[i])
        for i in range(len(edge_field_names)):
            self.add_edge_field(edge_field_names[i], edge_dtypes[i])
        for i in range(len(face_field_names)):
            self.add_face_field(face_field_names[i], face_dtypes[i])

    def del_center_field(self, field_name):
        del(self.center_fields[field_name])

    def del_edge_field(self, field_name):
        del(self.edge_fields[field_name])

    def del_face_field(self, field_name):
        del(self.face_fields[field_name])

    def swap_face_field(self, field_name_1, field_name_2):
        self.face_fields[field_name_1], self.face_fields[field_name_2] = self.face_fields[field_name_2], self.face_fields[field_name_1]

    def raise_mode_scalar_error(self, mode, scalar):
        if type(mode) != str:
            raise TypeError("Argument mode is 'str' not '%s'" % (str(type(mode)),))

        if mode not in ['set', '=', 'add', '+=']:
            raise ValueError("Invalid mode '%s'" % (mode,))

        if scalar == 0.0:
            return False

        return True

    def set_face_with_scalar(self, field_name, scalar):
        self.face_fields[field_name][:, :, :, :] = 0.0

    def set_edge_with_scalar(self, field_name, scalar):
        self.edge_fields[field_name][:, :, :, :] = 0.0

    def add_over_center(self, source_field, output_field, conj=False, mode = '+=', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.center_fields[output_field][:, :, :] = 0.0

            if conj:
                self.center_fields[output_field] += scalar * np.conj(self.center_fields[source_field])
            else:
                self.center_fields[output_field] += scalar*self.center_fields[source_field]

    def add_over_center_tensors(self, source_tensor_field, output_tensor_field, mode ='+=', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.center_tensor_fields[output_tensor_field][...] = 0.0

        self.center_tensor_fields[output_tensor_field][...] += scalar * self.center_tensor_fields[source_tensor_field]

    def add_over_center_tensor_component(self, source_field, output_tensor_field, i, j, mode ='+=', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.center_tensor_fields[output_tensor_field][i, j, ...] = 0.0

            for source_field_name in list(source_field):
                self.center_tensor_fields[output_tensor_field][i, j, ...] += scalar * self.center_fields[source_field_name]

    def add_over_face(self, source_field, output_field, mode = '+=', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.face_fields[output_field][...] = 0.0

            if scalar == 1.0:
                self.face_fields[output_field] += self.face_fields[source_field]
            elif scalar == -1.0:
                self.face_fields[output_field] -= self.face_fields[source_field]
            else:
                self.face_fields[output_field] += scalar*self.face_fields[source_field]

    def add_over_face_tensor(self, source_field, output_field, mode = '+=', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.face_tensor_fields[output_field][...] = 0.0

            self.face_tensor_fields[output_field][...] += scalar * self.face_tensor_fields[source_field]

    def add_over_edge(self, source_field, output_field, mode = '+=', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.edge_fields[output_field][:, :, :, :] = 0.0

            if scalar == 1.0:
                self.edge_fields[output_field] += self.edge_fields[source_field]
            elif scalar == -1.0:
                self.edge_fields[output_field] -= self.edge_fields[source_field]
            else:
                self.edge_fields[output_field] += scalar*self.edge_fields[source_field]

    def add_over_edge_tensor(self, source_field, output_field, mode = '+=', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.edge_tensor_fields[output_field][...] = 0.0

            self.edge_tensor_fields[output_field][...] += scalar * self.edge_tensor_fields[source_field]

    def mul_center_by_scalar(self, field_name, scalar):
        self.center_fields[field_name] *= scalar

    def mul_center_field_with_tensor(self, source_tensor_field, output_field, components, conj = False, scalar = 1.0):
            for cmp in list(components):
                if conj:
                    self.center_fields[output_field][...] *= scalar * np.conj(self.center_tensor_fields[source_tensor_field][cmp])
                else:
                    self.center_fields[output_field][...] *= scalar * self.center_tensor_fields[source_tensor_field][cmp]

    def mul_center_tensor_fields(self, source_field_1, source_field_2, output_field, mode = "+=", scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.center_tensor_fields[output_field][...] = 0.0

        for i in range(self.center_tensor_fields[output_field].shape[0]):
            for j in range(self.center_tensor_fields[output_field].shape[1]):
                for k in range(self.center_tensor_fields[source_field_1].shape[1]):
                    self.center_tensor_fields[output_field][i, j] += (scalar *
                                                                      self.center_tensor_fields[source_field_1][i, k] *
                                                                      self.center_tensor_fields[source_field_2][k, j])

    def mul_external_matrix_by_center_tensor(self, matrix, source_field, output_field, mode = '+=', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.center_tensor_fields[output_field][...] = 0.0

        for i in range(self.center_tensor_fields[output_field].shape[0]):
            for j in range(self.center_tensor_fields[output_field].shape[1]):
                for k in range(matrix.shape[1]):
                    self.center_tensor_fields[output_field][i, j] += (scalar * matrix[i, k] *
                                                                      self.center_tensor_fields[source_field][k, j])

    def mul_face_by_scalar(self, field_name, scalar):
        self.face_fields[field_name] *= scalar

    def mul_face_tensor_fields(self, source_field_1, source_field_2, output_field, mode = "+=", scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.face_tensor_fields[output_field][...] = 0.0

        for i in range(self.face_tensor_fields[output_field].shape[0]):
            for j in range(self.face_tensor_fields[output_field].shape[1]):
                for k in range(self.face_tensor_fields[source_field_1].shape[1]):
                    self.face_tensor_fields[output_field][i, j] += (scalar *
                                                                      self.face_tensor_fields[source_field_1][i, k] *
                                                                      self.face_tensor_fields[source_field_2][k, j])

    def mul_external_matrix_by_face_tensor(self, matrix, source_field, output_field, mode = '+=', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.face_tensor_fields[output_field][...] = 0.0

        for i in range(self.face_tensor_fields[output_field].shape[0]):
            for j in range(self.face_tensor_fields[output_field].shape[1]):
                for k in range(matrix.shape[1]):
                    self.face_tensor_fields[output_field][i, j] += (scalar * matrix[i, k] *
                                                                      self.face_tensor_fields[source_field][k, j])

    def diff_center_field(self, source_field, output_field, axis, mode = 'set', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.center_fields[output_field][:, :, :] = 0.0

            self.center_fields[output_field] += (0.5/self.spacings[axis]) * scalar * (np.roll(self.center_fields[source_field], -1, axis) - np.roll(self.center_fields[source_field], 1, axis))

    def diff2_center_field(self, source_field, output_field, axis, mode = 'set', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.center_fields[output_field][:, :, :] = 0.0

            self.center_fields[output_field] += (scalar/self.spacings[axis]**2) * (np.roll(self.center_fields[source_field], -1, axis) - 2.0*self.center_fields[source_field] + np.roll(self.center_fields[source_field], 1, axis))

    def lapl_center_field(self, source_field, output_field, mode = 'set', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.center_fields[output_field][:, :, :] = 0.0

            self.diff2_center_field(source_field, output_field, 0, mode, scalar)
            self.diff2_center_field(source_field, output_field, 1, '+=', scalar)
            self.diff2_center_field(source_field, output_field, 2, '+=', scalar)

    def curl_center(self, source_fieldX, source_fieldY, source_fieldZ, output_fieldX, output_fieldY, output_fieldZ, mode = 'set', scalar = 1.0):
        self.diff_center_field(source_fieldZ, output_fieldX, 1, mode, scalar)
        self.diff_center_field(source_fieldY, output_fieldX, 2, '+=', -scalar)

        self.diff_center_field(source_fieldX, output_fieldY, 2, mode, scalar)
        self.diff_center_field(source_fieldZ, output_fieldY, 0, '+=', -scalar)

        self.diff_center_field(source_fieldY, output_fieldZ, 0, mode, scalar)
        self.diff_center_field(source_fieldX, output_fieldZ, 1, '+=', -scalar)

    def curl_center_FFT(self, source_fieldX, source_fieldY, source_fieldZ, output_fieldX, output_fieldY, output_fieldZ, mode = 'set', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            kx, ky, kz = np.meshgrid(2.0*np.pi*np.fft.fftfreq(self.num_cells[0], self.spacings[0]),
                                     2.0*np.pi*np.fft.fftfreq(self.num_cells[1], self.spacings[1]),
                                     2.0*np.pi*np.fft.fftfreq(self.num_cells[2], self.spacings[2]),
                                     indexing = 'ij', copy = False)

            if mode in ['set', '=']:
                self.center_fields[output_fieldX][:, :, :] = 0.0
                self.center_fields[output_fieldY][:, :, :] = 0.0
                self.center_fields[output_fieldZ][:, :, :] = 0.0

            self.center_fields[output_fieldX] += scalar*np.fft.ifftn(1.0j*(-kz*np.fft.fftn(self.center_fields[source_fieldY]) + ky*np.fft.fftn(self.center_fields[source_fieldZ]))).real
            self.center_fields[output_fieldY] += scalar*np.fft.ifftn(1.0j*(-kx*np.fft.fftn(self.center_fields[source_fieldZ]) + kz*np.fft.fftn(self.center_fields[source_fieldX]))).real
            self.center_fields[output_fieldZ] += scalar*np.fft.ifftn(1.0j*(-ky*np.fft.fftn(self.center_fields[source_fieldX]) + kx*np.fft.fftn(self.center_fields[source_fieldY]))).real

    def div_center(self, source_fieldX, source_fieldY, source_fieldZ, output_field, mode = "set", scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.center_fields[output_field][:, :, :] = 0.0

            self.diff_center_field(source_fieldX, output_field, 0, "+=")
            self.diff_center_field(source_fieldY, output_field, 1, "+=")
            self.diff_center_field(source_fieldZ, output_field, 2, "+=")
            self.center_fields[output_field] *= scalar

    def div_face_to_center(self, source_field, output_field, mode ="set", scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.center_fields[output_field][:, :, :] = 0.0

            self.center_fields[output_field] += ( np.roll(self.face_fields[source_field], -1, 0) - self.face_fields[source_field] )/self.spacings[0]
            self.center_fields[output_field] += ( np.roll(self.face_fields[source_field], -1, 1) - self.face_fields[source_field] )/self.spacings[1]
            self.center_fields[output_field] += ( np.roll(self.face_fields[source_field], -1, 2) - self.face_fields[source_field] )/self.spacings[2]
            self.center_fields[output_field] *= scalar

    def grad_center_to_face(self, source_field, output_field, mode = "set", scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ["set", "="]:
                self.face_fields[output_field][:] = 0.0

            self.face_fields[output_field][0] += (scalar / self.spacings[0]) * (self.center_fields[source_field] - np.roll(self.center_fields[source_field], 1, 0))
            self.face_fields[output_field][1] += (scalar / self.spacings[1]) * (self.center_fields[source_field] - np.roll(self.center_fields[source_field], 1, 1))
            self.face_fields[output_field][2] += (scalar / self.spacings[2]) * (self.center_fields[source_field] - np.roll(self.center_fields[source_field], 1, 2))

    def curl_face_to_edge(self, source_field, output_field, mode = "set", scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.edge_fields[output_field][0][:, :, :] = 0.0
                self.edge_fields[output_field][1][:, :, :] = 0.0
                self.edge_fields[output_field][2][:, :, :] = 0.0

            self.edge_fields[output_field][0] += scalar * ((self.face_fields[source_field][2] - np.roll(self.face_fields[source_field][2], 1, 1)) / self.spacings[1] + (np.roll(self.face_fields[source_field][1], 1, 2) - self.face_fields[source_field][1]) / self.spacings[2])
            self.edge_fields[output_field][1] += scalar * ((self.face_fields[source_field][0] - np.roll(self.face_fields[source_field][0], 1, 2)) / self.spacings[2] + (np.roll(self.face_fields[source_field][2], 1, 0) - self.face_fields[source_field][2]) / self.spacings[0])
            self.edge_fields[output_field][2] += scalar * ((self.face_fields[source_field][1] - np.roll(self.face_fields[source_field][1], 1, 0)) / self.spacings[0] + (np.roll(self.face_fields[source_field][0], 1, 1) - self.face_fields[source_field][0]) / self.spacings[1])

    def curl_edge_to_face(self, source_field, output_field, mode = "set", scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.face_fields[output_field][0][:, :, :] = 0.0
                self.face_fields[output_field][1][:, :, :] = 0.0
                self.face_fields[output_field][2][:, :, :] = 0.0

            self.face_fields[output_field][0] += scalar * ((np.roll(self.edge_fields[source_field][2], -1, 1) -
                                                            self.edge_fields[source_field][2])
                                                           / self.spacings[1] +
                                                           (self.edge_fields[source_field][1] -
                                                            np.roll(self.edge_fields[source_field][1], -1, 2))
                                                           / self.spacings[2])

            self.face_fields[output_field][1] += scalar * ((np.roll(self.edge_fields[source_field][0], -1, 2) -
                                                            self.edge_fields[source_field][0])
                                                           / self.spacings[2] +
                                                           (self.edge_fields[source_field][2] -
                                                            np.roll(self.edge_fields[source_field][2], -1, 0))
                                                           / self.spacings[0])

            self.face_fields[output_field][2] += scalar * ((np.roll(self.edge_fields[source_field][1], -1, 0) -
                                                            self.edge_fields[source_field][1])
                                                           / self.spacings[0] +
                                                           (self.edge_fields[source_field][0] -
                                                            np.roll(self.edge_fields[source_field][0], -1, 1))
                                                           / self.spacings[1])

    def curl(self, source_field, source_type, output_field, output_type, mode = 'set', scalar = 1.0):
        if source_type == 'edge':
            if output_type == 'face':
                self.curl_edge_to_face(source_field, output_field, mode, scalar)

        if source_type == 'face':
            if output_type == 'edge':
                self.curl_face_to_edge(source_field, output_field, mode, scalar)

    def interpolate_face_to_center(self, source_field, output_fieldX, output_fieldY, output_fieldZ, mode = 'set', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.center_fields[output_fieldX][:, :, :] = 0.0
                self.center_fields[output_fieldY][:, :, :] = 0.0
                self.center_fields[output_fieldZ][:, :, :] = 0.0

            self.center_fields[output_fieldX] += 0.5*scalar*(self.face_fields[source_field][0] + np.roll(self.face_fields[source_field][0], -1, 0))
            self.center_fields[output_fieldY] += 0.5*scalar*(self.face_fields[source_field][1] + np.roll(self.face_fields[source_field][1], -1, 1))
            self.center_fields[output_fieldZ] += 0.5*scalar*(self.face_fields[source_field][2] + np.roll(self.face_fields[source_field][2], -1, 2))

    def interpolate_face_to_center_vector(self, source_field, output_field, mode = 'set', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.center_vector_fields[output_field][...] = 0.0

            self.center_vector_fields[output_field][0, ...] += 0.5*scalar*(self.face_fields[source_field][0] + np.roll(self.face_fields[source_field][0], -1, 0))
            self.center_vector_fields[output_field][1, ...] += 0.5*scalar*(self.face_fields[source_field][1] + np.roll(self.face_fields[source_field][1], -1, 1))
            self.center_vector_fields[output_field][2, ...] += 0.5*scalar*(self.face_fields[source_field][2] + np.roll(self.face_fields[source_field][2], -1, 2))

    def interpolate_edge_to_center(self, source_field, output_fieldX, output_fieldY, output_fieldZ, mode = 'set', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.center_fields[output_fieldX][:, :, :] = 0.0
                self.center_fields[output_fieldY][:, :, :] = 0.0
                self.center_fields[output_fieldZ][:, :, :] = 0.0

            self.center_fields[output_fieldX] = 0.25 * scalar * (self.face_fields[source_field][0] +
                                                                 multi_roll(self.face_fields[source_field][0], (0, 1, 0)) +
                                                                 multi_roll(self.face_fields[source_field][0], (0, 0, 1)) +
                                                                 multi_roll(self.face_fields[source_field][0], (0, 1, 1)))
            self.center_fields[output_fieldY] = 0.25 * scalar * (self.face_fields[source_field][1] +
                                                                 multi_roll(self.face_fields[source_field][1], (1, 0, 0)) +
                                                                 multi_roll(self.face_fields[source_field][1], (0, 0, 1)) +
                                                                 multi_roll(self.face_fields[source_field][1], (1, 0, 1)))
            self.center_fields[output_fieldZ] = 0.25 * scalar * (self.face_fields[source_field][2] +
                                                                 multi_roll(self.face_fields[source_field][2], (1, 0, 0)) +
                                                                 multi_roll(self.face_fields[source_field][2], (0, 1, 0)) +
                                                                 multi_roll(self.face_fields[source_field][2], (1, 1, 0)))

    def interpolate_center_to_face(self, source_fieldX, source_fieldY, source_fieldZ, output_field, mode = 'set', scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ['set', '=']:
                self.center_fields[output_field][...] = 0.0

            self.face_fields[output_field][0] += 0.5*scalar*(self.center_fields[source_fieldX] + np.roll(self.center_fields[source_fieldX], -1, 0))
            self.face_fields[output_field][1] += 0.5*scalar*(self.center_fields[source_fieldY] + np.roll(self.center_fields[source_fieldY], -1, 1))
            self.face_fields[output_field][2] += 0.5*scalar*(self.center_fields[source_fieldZ] + np.roll(self.center_fields[source_fieldZ], -1, 2))

    def interpolate_edge_scipy(self, field, x, y, z):
        points = np.array([x, y, z])
        points = np.rollaxis(points, 0, points.ndim)
        points.shape = (np.prod(x.shape), 3)

        Fx = RegularGridInterpolator((self.edgeX[0][:, 0, 0], self.edgeX[1][0, :, 0], self.edgeX[2][0, 0, :]), self.edge_fields[field][0], bounds_error=False)
        Fy = RegularGridInterpolator((self.edgeY[0][:, 0, 0], self.edgeY[1][0, :, 0], self.edgeY[2][0, 0, :]), self.edge_fields[field][1], bounds_error=False)
        Fz = RegularGridInterpolator((self.edgeZ[0][:, 0, 0], self.edgeZ[1][0, :, 0], self.edgeZ[2][0, 0, :]), self.edge_fields[field][2], bounds_error=False)
        retX = Fx(points)
        retY = Fy(points)
        retZ = Fz(points)
        retX.shape = x.shape
        retY.shape = x.shape
        retZ.shape = x.shape
        return np.array([retX, retY, retZ])

    def interpolate_face_scipy(self, field, x, y, z):
        points = np.array([x, y, z])
        points = np.rollaxis(points, 0, points.ndim)
        points.shape = (np.prod(x.shape), 3)

        Fx = RegularGridInterpolator((self.faceX[0][:, 0, 0], self.faceX[1][0, :, 0], self.faceX[2][0, 0, :]), self.face_fields[field][0], bounds_error = False)
        Fy = RegularGridInterpolator((self.faceY[0][:, 0, 0], self.faceY[1][0, :, 0], self.faceY[2][0, 0, :]), self.face_fields[field][1], bounds_error = False)
        Fz = RegularGridInterpolator((self.faceZ[0][:, 0, 0], self.faceZ[1][0, :, 0], self.faceZ[2][0, 0, :]), self.face_fields[field][2], bounds_error = False)
        retX = Fx(points)
        retY = Fy(points)
        retZ = Fz(points)
        retX.shape = x.shape
        retY.shape = x.shape
        retZ.shape = x.shape
        return np.array([retX, retY, retZ])

    def interpolate_center(self, field, X, Y, Z):
        dx = self.spacings[0]
        dy = self.spacings[1]
        dz = self.spacings[2]
        Nx = self.center.shape[1]
        Ny = self.center.shape[2]
        Nz = self.center.shape[3]
        Lx = Nx * dx
        Ly = Ny * dy
        Lz = Nz * dz

        x = X - 0.5 * dx
        y = Y - 0.5 * dy
        z = Z - 0.5 * dz

        idx1 = np.int32(x // dx) % Nx
        idx2 = np.int32(x // dx + 1) % Nx
        idy1 = np.int32(y // dy) % Ny
        idy2 = np.int32(y // dy + 1) % Ny
        idz1 = np.int32(z // dz) % Nz
        idz2 = np.int32(z // dz + 1) % Nz

        f111 = self.center_fields[field][idx1, idy1, idz1]
        f211 = self.center_fields[field][idx2, idy1, idz1]
        f121 = self.center_fields[field][idx1, idy2, idz1]
        f112 = self.center_fields[field][idx1, idy1, idz2]
        f221 = self.center_fields[field][idx2, idy2, idz1]
        f212 = self.center_fields[field][idx2, idy1, idz2]
        f122 = self.center_fields[field][idx1, idy2, idz2]
        f222 = self.center_fields[field][idx2, idy2, idz2]

        idx1 = x - idx1 * dx
        idx2 = idx2 * dx - x
        idy1 = y - idy1 * dy
        idy2 = idy2 * dy - y
        idz1 = z - idz1 * dz
        idz2 = idz2 * dz - z

        return (f111 * idx2 * idy2 * idz2 +
                f211 * idx1 * idy2 * idz2 +
                f121 * idx2 * idy1 * idz2 +
                f112 * idx2 * idy2 * idz1 +
                f221 * idx1 * idy1 * idz2 +
                f212 * idx1 * idy2 * idz1 +
                f122 * idx2 * idy1 * idz1 +
                f222 * idx1 * idy1 * idz1) / (dx * dy * dz)

    def interpolate_edgeX(self, field, X, Y, Z):
        dx = self.spacings[0]
        dy = self.spacings[1]
        dz = self.spacings[2]
        Nx = self.center.shape[1]
        Ny = self.center.shape[2]
        Nz = self.center.shape[3]
        Lx = Nx * dx
        Ly = Ny * dy
        Lz = Nz * dz

        x = X - 0.5*dx
        y = Y
        z = Z

        idx1 = np.int32(x // dx) % Nx
        idx2 = np.int32(x // dx + 1) % Nx
        idy1 = np.int32(y // dy) % Ny
        idy2 = np.int32(y // dy + 1) % Ny
        idz1 = np.int32(z // dz) % Nz
        idz2 = np.int32(z // dz + 1) % Nz

        f111 = self.edge_fields[field][0][idx1, idy1, idz1]
        f211 = self.edge_fields[field][0][idx2, idy1, idz1]
        f121 = self.edge_fields[field][0][idx1, idy2, idz1]
        f112 = self.edge_fields[field][0][idx1, idy1, idz2]
        f221 = self.edge_fields[field][0][idx2, idy2, idz1]
        f212 = self.edge_fields[field][0][idx2, idy1, idz2]
        f122 = self.edge_fields[field][0][idx1, idy2, idz2]
        f222 = self.edge_fields[field][0][idx2, idy2, idz2]

        idx1 = x - idx1 * dx
        idx2 = idx2 * dx - x
        idy1 = y - idy1 * dy
        idy2 = idy2 * dy - y
        idz1 = z - idz1 * dz
        idz2 = idz2 * dz - z

        return (f111 * idx2 * idy2 * idz2 +
                f211 * idx1 * idy2 * idz2 +
                f121 * idx2 * idy1 * idz2 +
                f112 * idx2 * idy2 * idz1 +
                f221 * idx1 * idy1 * idz2 +
                f212 * idx1 * idy2 * idz1 +
                f122 * idx2 * idy1 * idz1 +
                f222 * idx1 * idy1 * idz1) / (dx * dy * dz)

    def interpolate_edgeY(self, field, X, Y, Z):
        dx = self.spacings[0]
        dy = self.spacings[1]
        dz = self.spacings[2]
        Nx = self.center.shape[1]
        Ny = self.center.shape[2]
        Nz = self.center.shape[3]
        Lx = Nx * dx
        Ly = Ny * dy
        Lz = Nz * dz

        x = X
        y = Y - 0.5 * dy
        z = Z

        idx1 = np.int32(x // dx) % Nx
        idx2 = np.int32(x // dx + 1) % Nx
        idy1 = np.int32(y // dy) % Ny
        idy2 = np.int32(y // dy + 1) % Ny
        idz1 = np.int32(z // dz) % Nz
        idz2 = np.int32(z // dz + 1) % Nz

        f111 = self.edge_fields[field][1][idx1, idy1, idz1]
        f211 = self.edge_fields[field][1][idx2, idy1, idz1]
        f121 = self.edge_fields[field][1][idx1, idy2, idz1]
        f112 = self.edge_fields[field][1][idx1, idy1, idz2]
        f221 = self.edge_fields[field][1][idx2, idy2, idz1]
        f212 = self.edge_fields[field][1][idx2, idy1, idz2]
        f122 = self.edge_fields[field][1][idx1, idy2, idz2]
        f222 = self.edge_fields[field][1][idx2, idy2, idz2]

        idx1 = x - idx1 * dx
        idx2 = idx2 * dx - x
        idy1 = y - idy1 * dy
        idy2 = idy2 * dy - y
        idz1 = z - idz1 * dz
        idz2 = idz2 * dz - z

        return (f111 * idx2 * idy2 * idz2 +
                f211 * idx1 * idy2 * idz2 +
                f121 * idx2 * idy1 * idz2 +
                f112 * idx2 * idy2 * idz1 +
                f221 * idx1 * idy1 * idz2 +
                f212 * idx1 * idy2 * idz1 +
                f122 * idx2 * idy1 * idz1 +
                f222 * idx1 * idy1 * idz1) / (dx * dy * dz)

    def interpolate_edgeZ(self, field, X, Y, Z):
        dx = self.spacings[0]
        dy = self.spacings[1]
        dz = self.spacings[2]
        Nx = self.center.shape[1]
        Ny = self.center.shape[2]
        Nz = self.center.shape[3]
        Lx = Nx * dx
        Ly = Ny * dy
        Lz = Nz * dz

        x = X
        y = Y
        z = Z - 0.5 * dz

        idx1 = np.int32(x // dx) % Nx
        idx2 = np.int32(x // dx + 1) % Nx
        idy1 = np.int32(y // dy) % Ny
        idy2 = np.int32(y // dy + 1) % Ny
        idz1 = np.int32(z // dz) % Nz
        idz2 = np.int32(z // dz + 1) % Nz

        f111 = self.edge_fields[field][2][idx1, idy1, idz1]
        f211 = self.edge_fields[field][2][idx2, idy1, idz1]
        f121 = self.edge_fields[field][2][idx1, idy2, idz1]
        f112 = self.edge_fields[field][2][idx1, idy1, idz2]
        f221 = self.edge_fields[field][2][idx2, idy2, idz1]
        f212 = self.edge_fields[field][2][idx2, idy1, idz2]
        f122 = self.edge_fields[field][2][idx1, idy2, idz2]
        f222 = self.edge_fields[field][2][idx2, idy2, idz2]

        idx1 = x - idx1 * dx
        idx2 = idx2 * dx - x
        idy1 = y - idy1 * dy
        idy2 = idy2 * dy - y
        idz1 = z - idz1 * dz
        idz2 = idz2 * dz - z

        return (f111 * idx2 * idy2 * idz2 +
                f211 * idx1 * idy2 * idz2 +
                f121 * idx2 * idy1 * idz2 +
                f112 * idx2 * idy2 * idz1 +
                f221 * idx1 * idy1 * idz2 +
                f212 * idx1 * idy2 * idz1 +
                f122 * idx2 * idy1 * idz1 +
                f222 * idx1 * idy1 * idz1) / (dx * dy * dz)

    def interpolate_edge(self, field, X, Y, Z):
        return np.array([self.interpolate_edgeX(field, X, Y, Z),
                         self.interpolate_edgeY(field, X, Y, Z),
                         self.interpolate_edgeZ(field, X, Y, Z)])

    def interpolate_faceX(self, field, X, Y, Z):
        dx = self.spacings[0]
        dy = self.spacings[1]
        dz = self.spacings[2]
        Nx = self.center.shape[1]
        Ny = self.center.shape[2]
        Nz = self.center.shape[3]
        Lx = Nx * dx
        Ly = Ny * dy
        Lz = Nz * dz

        x = X
        y = Y - 0.5*dy
        z = Z - 0.5*dz

        idx1 = np.int32(x // dx) % Nx
        idx2 = np.int32(x // dx + 1) % Nx
        idy1 = np.int32(y // dy) % Ny
        idy2 = np.int32(y // dy + 1) % Ny
        idz1 = np.int32(z // dz) % Nz
        idz2 = np.int32(z // dz + 1) % Nz

        f111 = self.face_fields[field][0][idx1, idy1, idz1]
        f211 = self.face_fields[field][0][idx2, idy1, idz1]
        f121 = self.face_fields[field][0][idx1, idy2, idz1]
        f112 = self.face_fields[field][0][idx1, idy1, idz2]
        f221 = self.face_fields[field][0][idx2, idy2, idz1]
        f212 = self.face_fields[field][0][idx2, idy1, idz2]
        f122 = self.face_fields[field][0][idx1, idy2, idz2]
        f222 = self.face_fields[field][0][idx2, idy2, idz2]

        idx1 = x - idx1 * dx
        idx2 = idx2 * dx - x
        idy1 = y - idy1 * dy
        idy2 = idy2 * dy - y
        idz1 = z - idz1 * dz
        idz2 = idz2 * dz - z

        return (f111 * idx2 * idy2 * idz2 +
                f211 * idx1 * idy2 * idz2 +
                f121 * idx2 * idy1 * idz2 +
                f112 * idx2 * idy2 * idz1 +
                f221 * idx1 * idy1 * idz2 +
                f212 * idx1 * idy2 * idz1 +
                f122 * idx2 * idy1 * idz1 +
                f222 * idx1 * idy1 * idz1) / (dx * dy * dz)

    def interpolate_faceY(self, field, X, Y, Z):
        dx = self.spacings[0]
        dy = self.spacings[1]
        dz = self.spacings[2]
        Nx = self.center.shape[1]
        Ny = self.center.shape[2]
        Nz = self.center.shape[3]
        Lx = Nx * dx
        Ly = Ny * dy
        Lz = Nz * dz

        x = X - 0.5 * dx
        y = Y
        z = Z - 0.5 * dz

        idx1 = np.int32(x // dx) % Nx
        idx2 = np.int32(x // dx + 1) % Nx
        idy1 = np.int32(y // dy) % Ny
        idy2 = np.int32(y // dy + 1) % Ny
        idz1 = np.int32(z // dz) % Nz
        idz2 = np.int32(z // dz + 1) % Nz

        f111 = self.face_fields[field][1][idx1, idy1, idz1]
        f211 = self.face_fields[field][1][idx2, idy1, idz1]
        f121 = self.face_fields[field][1][idx1, idy2, idz1]
        f112 = self.face_fields[field][1][idx1, idy1, idz2]
        f221 = self.face_fields[field][1][idx2, idy2, idz1]
        f212 = self.face_fields[field][1][idx2, idy1, idz2]
        f122 = self.face_fields[field][1][idx1, idy2, idz2]
        f222 = self.face_fields[field][1][idx2, idy2, idz2]

        idx1 = x - idx1 * dx
        idx2 = idx2 * dx - x
        idy1 = y - idy1 * dy
        idy2 = idy2 * dy - y
        idz1 = z - idz1 * dz
        idz2 = idz2 * dz - z

        return (f111 * idx2 * idy2 * idz2 +
                f211 * idx1 * idy2 * idz2 +
                f121 * idx2 * idy1 * idz2 +
                f112 * idx2 * idy2 * idz1 +
                f221 * idx1 * idy1 * idz2 +
                f212 * idx1 * idy2 * idz1 +
                f122 * idx2 * idy1 * idz1 +
                f222 * idx1 * idy1 * idz1) / (dx * dy * dz)

    def interpolate_faceZ(self, field, X, Y, Z):
        dx = self.spacings[0]
        dy = self.spacings[1]
        dz = self.spacings[2]
        Nx = self.center.shape[1]
        Ny = self.center.shape[2]
        Nz = self.center.shape[3]
        Lx = Nx * dx
        Ly = Ny * dy
        Lz = Nz * dz

        x = X - 0.5 * dx
        y = Y - 0.5 * dy
        z = Z

        idx1 = np.int32(x // dx) % Nx
        idx2 = np.int32(x // dx + 1) % Nx
        idy1 = np.int32(y // dy) % Ny
        idy2 = np.int32(y // dy + 1) % Ny
        idz1 = np.int32(z // dz) % Nz
        idz2 = np.int32(z // dz + 1) % Nz

        f111 = self.face_fields[field][2][idx1, idy1, idz1]
        f211 = self.face_fields[field][2][idx2, idy1, idz1]
        f121 = self.face_fields[field][2][idx1, idy2, idz1]
        f112 = self.face_fields[field][2][idx1, idy1, idz2]
        f221 = self.face_fields[field][2][idx2, idy2, idz1]
        f212 = self.face_fields[field][2][idx2, idy1, idz2]
        f122 = self.face_fields[field][2][idx1, idy2, idz2]
        f222 = self.face_fields[field][2][idx2, idy2, idz2]

        idx1 = x - idx1 * dx
        idx2 = idx2 * dx - x
        idy1 = y - idy1 * dy
        idy2 = idy2 * dy - y
        idz1 = z - idz1 * dz
        idz2 = idz2 * dz - z

        return (f111 * idx2 * idy2 * idz2 +
                f211 * idx1 * idy2 * idz2 +
                f121 * idx2 * idy1 * idz2 +
                f112 * idx2 * idy2 * idz1 +
                f221 * idx1 * idy1 * idz2 +
                f212 * idx1 * idy2 * idz1 +
                f122 * idx2 * idy1 * idz1 +
                f222 * idx1 * idy1 * idz1) / (dx * dy * dz)

    def interpolate_face(self, field, X, Y, Z):
        return np.array([self.interpolate_faceX(field, X, Y, Z),
                         self.interpolate_faceY(field, X, Y, Z),
                         self.interpolate_faceZ(field, X, Y, Z)])

    def poisson_solver_center_FFT(self, source_field, output_field):
        self.center_fields[output_field][:, :, :] = 0.0
        aux = np.fft.rfftn(self.center_fields[source_field])
        aux /= -np.sum(np.array(np.meshgrid(2.0*np.pi*np.fft.fftfreq(self.center.shape[1], self.spacings[0]),
                                            2.0*np.pi*np.fft.fftfreq(self.center.shape[2], self.spacings[1]),
                                            2.0*np.pi*np.fft.rfftfreq(self.center.shape[3], self.spacings[2]), indexing = 'ij', copy = False))**2, axis = 0)
        aux[0, 0, 0] = 0.0
        self.center_fields[output_field] += np.fft.irfftn(aux).real

    def poisson_solver_face(self, source_field, output_field, mode = "set", scalar = 1.0):
        if self.raise_mode_scalar_error(mode, scalar):
            if mode in ["set", "="]:
                self.face_fields[output_field][:, :, :, :] = 0.0

            W = np.meshgrid(2.0*np.pi*np.fft.fftfreq(self.center.shape[1], self.spacings[0]),
                            2.0*np.pi*np.fft.fftfreq(self.center.shape[2], self.spacings[1]),
                            2.0*np.pi*np.fft.fftfreq(self.center.shape[3], self.spacings[2]), indexing = 'ij', copy = False)

            for i in [0, 1, 2]:
                aux = np.fft.fftn(self.face_fields[source_field][i])
                aux /= -np.sum(np.array(W)**2, axis = 0)
                aux[0, 0, 0] = 0.0
                self.face_fields[output_field][i] += scalar*np.fft.ifftn(aux).real

    def poisson_solver_center_GM(self, source_field, output_field, aux_field, step = 10.0):
        dx2 = self.spacings[0]**2
        dy2 = self.spacings[1]**2
        dz2 = self.spacings[2]**2
        dx4 = self.spacings[0]**4
        dy4 = self.spacings[1]**4
        dz4 = self.spacings[2]**4
        f = np.zeros((2, 2, 2))
        phi = np.zeros((5, 5, 5))

        f[ 1,  0,  0] = 2.0/dx2
        f[-1,  0,  0] = 2.0/dx2
        f[ 0,  1,  0] = 2.0/dy2
        f[ 0, -1,  0] = 2.0/dy2
        f[ 0,  0, -1] = 2.0/dz2
        f[ 0,  0,  1] = 2.0/dz2

        f[ 0,  0,  0] = -4.0/dx2 - 4.0/dy2 - 4.0/dz2

        phi[ 2,  0,  0] = -2.0/dx4
        phi[-2,  0,  0] = -2.0/dx4
        phi[ 0,  2,  0] = -2.0/dy4
        phi[ 0, -2,  0] = -2.0/dy4
        phi[ 0,  0,  2] = -2.0/dz4
        phi[ 0,  0, -2] = -2.0/dz4

        phi[ 1,  0,  0] = 8.0*(1.0 + dx2*(1.0/dy2 + 1.0/dz2))/dx4
        phi[-1,  0,  0] = 8.0*(1.0 + dx2*(1.0/dy2 + 1.0/dz2))/dx4
        phi[ 0,  1,  0] = 8.0*(1.0 + dy2*(1.0/dx2 + 1.0/dz2))/dy4
        phi[ 0, -1,  0] = 8.0*(1.0 + dy2*(1.0/dx2 + 1.0/dz2))/dy4
        phi[ 0,  0,  1] = 8.0*(1.0 + dz2*(1.0/dx2 + 1.0/dy2))/dz4
        phi[ 0,  0, -1] = 8.0*(1.0 + dz2*(1.0/dx2 + 1.0/dy2))/dz4

        phi[-1, -1,  0] = -4.0/(dx2*dy2)
        phi[-1,  1,  0] = -4.0/(dx2*dy2)
        phi[ 1, -1,  0] = -4.0/(dx2*dy2)
        phi[ 1,  1,  0] = -4.0/(dx2*dy2)

        phi[ 0, -1, -1] = -4.0/(dy2*dz2)
        phi[ 0, -1,  1] = -4.0/(dy2*dz2)
        phi[ 0,  1, -1] = -4.0/(dy2*dz2)
        phi[ 0,  1,  1] = -4.0/(dy2*dz2)

        phi[-1,  0, -1] = -4.0/(dx2*dz2)
        phi[-1,  0,  1] = -4.0/(dx2*dz2)
        phi[ 1,  0, -1] = -4.0/(dx2*dz2)
        phi[ 1,  0,  1] = -4.0/(dx2*dz2)

        phi[ 0,  0,  0] = 12.0/dx4 + 12.0/dy4 + 12.0/dz4 + 16.0/(dy2*dz2) + (16.0/dy2 + 16.0/dz2)/dx2


        self.center_fields[output_field][:, :, :] = np.random.random(self.center.shape[1:])
        for iteration in range(10):
            print(iteration)
            self.center_fields[aux_field][:, :, :] = 0.0
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        if f[i, j, k] != 0.0:
                            self.center_fields[aux_field] += f[i, j, k] * np.roll(np.roll(np.roll(self.center_fields[source_field], i, 0), j, 1), k, 2)

            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        if phi[i, j, k] != 0.0:
                            self.center_fields[aux_field] += phi[i, j, k] * np.roll(np.roll(np.roll(self.center_fields[output_field], i, 0), j, 1), k, 2)

            self.center_fields[aux_field] *= step
            self.center_fields[output_field] += self.center_fields[aux_field]

    def poisson_solver_center_SM(self, source_field, output_field):
        lapl = lil_matrix((np.prod(self.center.shape[1:]),)*2, dtype = np.float64)

        Lx = self.center.shape[1]
        Ly = self.center.shape[2]
        Lz = self.center.shape[3]
        dx = self.spacings[0]
        dy = self.spacings[1]
        dz = self.spacings[2]

        for i in range(Lx):
            print(i)
            for j in range(Ly):
                for k in range(Lz):
                    lapl[idx(i, j, k, Lx, Ly, Lz), idx(i, j, k, Lx, Ly, Lz)] = -2.0 / dx ** 2 - 2.0 / dy ** 2 - 2.0 / dz ** 2
                    lapl[idx(i, j, k, Lx, Ly, Lz), idx(i - 1, j, k, Lx, Ly, Lz)] = 1.0 / dx ** 2
                    lapl[idx(i, j, k, Lx, Ly, Lz), idx(i + 1, j, k, Lx, Ly, Lz)] = 1.0 / dx ** 2
                    lapl[idx(i, j, k, Lx, Ly, Lz), idx(i, j - 1, k, Lx, Ly, Lz)] = 1.0 / dy ** 2
                    lapl[idx(i, j, k, Lx, Ly, Lz), idx(i, j + 1, k, Lx, Ly, Lz)] = 1.0 / dy ** 2
                    lapl[idx(i, j, k, Lx, Ly, Lz), idx(i, j, k - 1, Lx, Ly, Lz)] = 1.0 / dz ** 2
                    lapl[idx(i, j, k, Lx, Ly, Lz), idx(i, j, k + 1, Lx, Ly, Lz)] = 1.0 / dz ** 2

        lapl = lapl.asformat('csc')
        self.center_fields[source_field].shape = (np.prod(self.center.shape[1:]),)
        self.center_fields[output_field].shape = (np.prod(self.center.shape[1:]),)
        T = time.clock()
        self.center_fields[output_field] = spsolve(lapl, self.center_fields[source_field], permc_spec='MMD_AT_PLUS_A')
        print('time:', time.clock() - T)
        self.center_fields[source_field].shape = self.center.shape[1:]
        self.center_fields[output_field].shape = self.center.shape[1:]

    def poisson_solver_center_CG(self, source_field, output_field, aux_field_r, aux_field_r_1, aux_field_p, aux_field_Ap):
        self.center_fields[aux_field_r][:, :, :] = self.center_fields[source_field]
        self.lapl_center_field(output_field, aux_field_r, '+=', -1.0)
        self.center_fields[aux_field_p][:, :, :] = self.center_fields[aux_field_r][:, :, :]

        i = 0
        while i < 20:
            print(i)
            self.lapl_center_field(aux_field_p, aux_field_Ap)
            alpha = np.sum(self.center_fields[aux_field_r]**2) / np.sum(self.center_fields[aux_field_p]*self.center_fields[aux_field_Ap])
            self.center_fields[output_field] += alpha*self.center_fields[aux_field_p]
            self.center_fields[aux_field_r_1][:, :, :] = self.center_fields[aux_field_r] - alpha*self.center_fields[aux_field_Ap]

            beta = np.sum(self.center_fields[aux_field_r_1]**2) / np.sum(self.center_fields[aux_field_r]**2)

            self.center_fields[aux_field_p][:, :, :] = self.center_fields[aux_field_r_1] + beta * self.center_fields[aux_field_p]

            i += 1

    def poisson_solver_center_GD(self, source_field, output_field, aux_field):
        i = 0
        while i < 30:
            print(i)
            self.lapl_center_field(output_field, aux_field)
            self.center_fields[aux_field] -= self.center_fields[source_field]
            self.lapl_center_field(aux_field, output_field, '+=', 0.01)
            i += 1

    def poisson_solver_center(self, source_field, output_field, mode = "set", scalar = 1.0, solver = "FFT"):
        self.raise_mode_scalar_error(mode, scalar)
        self.poisson_solver_center_FFT(source_field, output_field, mode, scalar)