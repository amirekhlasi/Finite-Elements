import numpy as np

class point(object):
    def __init__(self, parameters):
        self.params = list(parameters)
        assert all([isinstance(p,float) for p in self.params])
        self.np_params = np.array(self.params)
        assert self.np_params.shape == (2,)

    def __eq__(self, other):
        if isinstance(other, point):
            if other.params == self.params:
                return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "point(" + repr(self.params) + ")"

    def __str__(self):
        return "point(" + str(self.params) + ")"

    def __hash__(self):
        return hash(repr(self))


class segment(object):
    def __init__(self, points):
        self.points = list(points)
        assert all([isinstance(p,point) for p in self.points])
        assert len(self.points) == 2
        self.params = [p.params for p in self.points]
        self.np_params = np.array(self.params)

    def reverse(self):
        return segment(reversed(self.points))

    def transport(self):
        return self.np_params[1] - self.np_params[0], self.np_params[0]

    def length(self):
        return ((self.np_params*np.array([[1],[-1]])).sum(0)**2).sum()

    def __eq__(self, other):
        if isinstance(other,segment):
            return self.points == other.points
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "segment(" + repr(self.params) + ")"

    def __str__(self):
        return "segment" + str(self.params) + ")"



class triangle(object):
    def __init__(self, vertices):
        self.points = list(vertices)
        assert all([isinstance(p,point) for p in self.points])
        assert len(self.points) == 3
        self.params = [p.params for p in self.points]
        self.np_params = np.array(self.params)



    def isinide(self, p0):
        a = self.np_params - p0.np_params
        b = [np.sign(np.linalg.det(a[[0,1],:])), np.sign(np.linalg.det(a[[1,2],:])), np.sign(np.linalg.det(a[[2,0],:]))]
        return all([x == 1 for x in b]) or any([x == 0 for x in b])

    def transport(self, p0):
        p0 = p0.np_params
        T = self.np_params[self.np_params != p0] - p0
        if np.linalg.det(T) < 0:
            T = T[[1,0]]
        return T.transpose(), p0

    def vertices(self):
        return list(self.points).copy()

    def rotate_vertices(self, rot = [0,1,2]):
        return triangle([point(p.params) for p in self.np_params[rot,:]])

    def size(self):
        return 0.5*np.abs(np.linalg.det(np.c_[self.np_params,np.ones(1)]))

    def __eq__(self, other):
        if not isinstance(other, triangle):
            return False
        return all([p in self.points for p in other.points])

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "triangle" + repr(self.points) + ")"

    def __str__(self):
        return "triangle" + str(self.points) + ")"
