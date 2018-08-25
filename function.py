import numpy as np
import geometry as gm


class function(object):

    def point_value(self,point):
        return None

    def eval(self, p):
        return None

    def segment_int(self, line, num = 0):
        return None

    def triangle_int(self, t, p0):
        return None

    def __repr__(self):
        return "None"

    def __str__(self):
        return "None"


class constant(function):
    def __init__(self, c):
        assert isinstance(c, float)
        self.c = c

    def eval(self, p):
        return self.c

    def point_value(self, point):
        return self.c

    def segment_int(self, line, num = 0):
        return self.c*line.length()

    def triangle_int(self, t, p0):
        return self.c*t.size()/3

    def __eq__(self, other):
        if isinstance(other, constant):
            return self.c == other.c
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return repr(self.c)

    def __str__(self):
        return str(self.c)

class zero(constant):
    def __init__(self):
        self.c = 0.0


class sin(function):

    def __init__(self, a, b, c):
        assert all([isinstance(x, float) for x in [a,b,c]])
        self.a = a
        self.b = b
        self.c = c
        self.vector = np.array([a,b])


    def eval(self,p):
        return np.sin(self.vector.dot(p.np_params) + self.c)

    def base_line_int(self, num = 0):
        if num == 1:
            return (-np.sin(self.a + self.c) + self.a*np.cos(self.a + self.c) + np.sin(self.c))/(self.a**2)
        if num == 0:
            return (-np.sin(self.a + self.c) + self.a*np.cos(self.c) + np.sin(self.c))/(self.a**2)

    def segment_int(self, line, num = 0):
        T, u = line.transport()
        return sin(self.vector.dot(T), 0, self.vector.dot(u) + self.c).base_line_int(num)*line.length()

    def base_triangle_int(self):
        return ((self.a**2 - self.b**2)*np.cos(c) + (self.b**2)*np.cos(self.a + self.c) -
                self.a*(self.b*(self.a - self.b)*np.sin(self.c) +
                        self.a*np.cos(self.b + self.c)))/(self.a**2*self.b**2*(self.a - self.b))

    def triangle_int(self, t, p0):
        T, u = t.transport(p0)
        v = self.vector.dot(T)
        c = self.vector.dot(u) + self.c
        return sin(v[0], v[1], c).base_triangle_int()*np.linalg.det(T)

    def __eq__(self, other):
        if isinstance(other, sin):
            return self.vector == other.vector && self.c == other.c
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "sin(" + str(self.a) + "x" + " + " + str(self.b) + "y" + " + " + str(self.c) + ")"

    def __str__(self):
        return "sin(" + str(self.a) + "x" + " + " + str(self.b) + "y" + " + " + str(self.c) + ")"



class element(function):
    def __init__(self, p0, triangles, boundary = []):
        assert isinstance(p0, gm.point)
        assert isinstance(triangles, list)
        assert all([isinstance(t, gm.triangle) for t in triangles])
        assert isinstance(boundary, list)
        assert all([isinstance(line, gm.segment) for line in boundary])
        assert all([p0 in line.points for line in boundary])
        assert all([p0 in t.points for t in triangles])
        self.triangles = triangles
        self.point = p0
        self.boundary0 = [b for b in boundary if b.points[0] == p0]
        self.boundary1 = [b for b in boundary if b.points[1] == p0]

    def eval(self, p):
        tr = [t for t in self.triangles if t.inside(p)]
        if len(tr) == 0:
            return 0.0
        t = tr[0]
        A = np.linalg.inv(t.transport(self.point)[0])
        return 1 - np.sum(A.dot(p.np_params - self.point))

    def grad(self, t):
        if t not in self.triangles:
            return np.array([0.0, 0.0])
        A = t.np_params[t.np_params != self.point.np_params] - self.point.np_params
        return np.linalg.inv(A).dot(np.array([1.0, 1.0]))

    def int(self, func):
        return sum([func.triangle_int(t, self.point) for t in self.triangles])

    def boundary_int(self, func):
        return sum([func.segmant_int(s, 0) for s in self.boundary0]) + sum([func.segmant_int(s, 1) for s in self.boundary1])

    def segment_int(self, line, num = 0):
        if self.point == line.points[0]:
            if num == 0:
                return line.length()/3
            if num == 1:
                return line.length()/6
        if self.point == line.points[1]:
            if num == 0:
                return line.length()/6
            if num == 1:
                return line.length()/3
        return 0

    def triangle_int(self, t, p0):
        if t not in self.triangles:
            return 0;
        if p0 == self.point:
            return t.size()/6
        return t.size()/12

    def grad_int(self, elem, t):
        triangles = [t for t in self.triangles if t in elem.triangles]
        return sum([self.grad(t).reshape(2,1).dot(elem.grad(t).reshape(1,2))*t.size() for t in triangles])

    def __eq__(self, other):
        if not isinstance(other, element):
            return False
        if self.point != element.point:
            return False
        return all([t in self.triangles for t in element.triangles]) and all([t in element.triangles for t in self.triangles])

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "element(point = " + repr(self.point) + ", triangles = [" + repr(self.triangles) + "])"

    def __str__(self):
        return "element(point = " + str(self.point) + ", triangles = [" + str(self.triangles) + "])"


class linear_comb(function):
    def __init__(self, combination, not_repeated = False):
        assert isinstance(combination, list)
        assert all([len(f) == 2 for f in combination])
        assert all([isinstance(f[1], function) and isinstance(f[0], float) for f in combination])
        self.functions = []
        self.coefficients = []
        if not not_repeated:
            for c, f in combination:
                if f not in self.functions and c != 0.0:
                    self.functions.append(f)
                    self.coefficients.append(c)
                elif c != 0:
                    i = self.functions.index(f)
                    self.coefficients[i] = self.coefficients[i] + c
        else:
            for c, f in combination:
                if c != 0:
                    self.functions.append(f)
                    self.coefficients.append(c)
        self.combination = list(zip(self.coefficients, self.combination))
        self.zero = len(self.combination) == 0

    def eval(self, p):
        return sum([c * f.eval(p) for c, f in self.combination])

    def segment_int(self, line, num = 0):
        return  sum([c * f.segment_int(line, num) for c, f in self.combination])

    def triangle_int(self, t, p0):
        return sum([c * f.triangle_int for c, f in self.combination])

    def __eq__(self, other):
        if not isinstance(other, linear_comb):
            return False
        if self.zero and other.zero:
            return True
        if self.zero and (not other.zero):
            return False
        if (not self.zero) and other.zero:
            return False
        if len(self.combination) != len(other.combination):
            return False
        if any([f not in other.functions for f in self.functions]) or\
                any([f not in self.functions for f in other.functions]):
            return False

        for i in range(len(self.functions) + 1):
            f = self.functions[i]
            j = other.functions.index(f)
            if self.coefficients[i] != other.coefficients[j]:
                return False
        return True
