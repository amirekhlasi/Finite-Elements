import geometry as gm
import numpy as np

class triangulation(object):
    def __init__(self):
        self.points = []
        self.triangles = []
        self.boundary = []
        self.point_boundary = {}
        self.point_triangles = {}
        self.point_dictionary = {}

    def inside(self, p):
        return False

    def triangle(self, p):
        for t in self.triangles:
            if t.isinside(p):
                return t
        return None

    def __repr__(self):
        return "triangulation(points = " + repr(self.points) + ", triangles =" + repr(self.triangles) + ")"

    def __str__(self):
        return "triangulation(points = " + str(self.points) + ", triangles =" + str(self.triangles) + ")"



class rectangle_triangulation(triangulation):
    def __init__(self, rec, x_mesh_num, y_mesh_num):
        rec = tuple(rec)
        assert isinstance(rec, tuple)
        assert all([isinstance(p, gm.point) for p in rec])
        assert all(rec[0].np_params < rec[1].np_params)
        assert all([isinstance(x, int) and x > 2 for x in [x_mesh_num, y_mesh_num]])
        super().__init__()
        self.rec = rec
        self.x_mesh_num = x_mesh_num
        self.y_mesh_num = y_mesh_num
        self.a, self.b = [x.params for x in self.rec]
        self.x_mesh = (self.b[0] - self.a[0])/self.x_mesh_num
        self.y_mesh = (self.b[1] - self.a[1])/self.y_mesh_num
        # skeleton_points
        self.skeleton_points = []
        for i in range(self.x_mesh_num + 1):
            skl = []
            self.skeleton_points.append(skl)
            for j in range(self.y_mesh_num + 1):
                p = gm.point([i*self.x_mesh, j*self.x_mesh])
                self.points.append(p)
                self.point_boundary[p] = []
                self.point_triangles[p] = []
                skl.append(p)
        # triangles and mid points
        self.mid_points = []
        for i in range(self.x_mesh_num):
            mid = []
            self.mid_points.append(mid)
            for j in range(self.y_mesh_num):
                p1, p2, p3, p4 = self.skeleton_points[i][j], self.skeleton_points[i + 1][j],\
                                 self.skeleton_points[i][j + 1], self.skeleton_points[i + 1][j + 1]
                params = (p1.np_params + p4.np_params)/2
                q = gm.point(list(params))
                mid.append(q)
                self.points.append(q)
                self.point_triangles[q] = []
                self.point_boundary[q] = []
                for l in [[p1, p2, q], [p1, p3, q], [p3, p4, q], [p2, p4, q]]:
                    t = gm.triangle(l)
                    self.triangles.append(t)
                    for p in l:
                        self.point_triangles[p].append(t)
        # boundry
        row1 = self.skeleton_points[0]
        row2 = self.skeleton_points[self.y_mesh_num][::-1]
        col1 = [row[0] for row in self.skeleton_points[::-1]]
        col2 = [row[self.x_mesh_num] for row in self.skeleton_points]
        boundary = [row1, col2, row2, col1]
        for row in boundary:
            for i in range(len(row) - 1):
                l = [row[i], row[i + 1]]
                seg = gm.segment(l)
                self.boundary.append(seg)
                for p in l:
                    self.point_boundary[p].append(seg)
        # dictionary
        self.point_dictionary = {p: i for i, p in enumerate(self.points)}

    def inside(self, p):
        if (self.rec[0].np_params <= p.np_params).all() and (self.rec[1].np_params >= p.np_params).all():
            return True
        return False











