import numpy as np
from scipy.sparse import linalg as lsolver
import geometry as gm
import function as fc
import triangulation

class solve(object):
    def __init__(self, tr, main_function, boundary_function, main_coeficient = 1.0 , laplacian_coeficients = np.identity(2, float)):
        assert isinstance(tr, triangulation.triangulation)
        assert isinstance(main_function, fc.function)
        assert isinstance(boundary_function, fc.function)
        assert isinstance(main_coeficient, float)
        self.laplacian_coeffients = np.array(laplacian_coeficients)
        assert self.laplacian_coeffients.shape == (2,2)
        assert np.allclose(self.laplacian_coeffients, self.laplacian_coeffients.T, atol=1e-8)
        self.main_coefficient = main_coeficient
        self.tr = tr
        self.main_function = main_function
        self.boundary_function = boundary_function
        self.point_elements = {p: fc.element(p, self.tr.point_triangles[p], self.tr.point_boundary[p]) for p in self.tr.points}
        self.elements = [self.point_elements[p] for p in self.tr.points]
        self.num = len(self.tr.points)
        self.__make__()
        self.__solve__()

    def __make__(self):
        self.main_res_vector = np.zeros(self.num, float)
        self.boundary_res_vector = np.zeros(self.num, float)
        self.grad_matrix = np.zeros((self.num, self.num), float)
        self.main_matrix = np.zeros((self.num, self.num), float)
        for i in range(self.num):
            elem1 = self.elements[i]
            self.main_res_vector[i] = elem1.int(self.main_function)
            self.boundary_res_vector[i] = elem1.int(self.boundary_function)
            for j in range(i + 1):
                elem2 = self.elements[j]
                self.main_matrix[i, j] = self.main_matrix[j, i] = fc.element.int(elem1, elem2)*self.main_coefficient
                self.grad_matrix[i, j] = self.grad_matrix[j, i] = (fc.element.grad_int(elem1, elem2) *
                                                                   self.laplacian_coeffients).sum()

    def __solve__(self):
        A = self.main_matrix + self.grad_matrix
        b = self.main_res_vector + self.boundary_res_vector
        self.coefficients, self.res = lsolver.bicg(A, b)
        self.coefficients = np.array(self.coefficients)

    def eval(self, p):
        if not self.tr.inside(p):
            return 0.0
        t = self.tr.triangle(p)
        if t is None:
            return 0.0
        x = self.coefficients[[self.tr.point_dictionary[p] for p in t.points]]
        a = (t.np_params - p.np_params)**2
        c = a.sum(1) / a.sum()
        return np.sum(c*x)

    def export_function(self):
        return list(zip(list(self.coefficients), self.elements))






