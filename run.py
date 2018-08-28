import geometry as gm
import function as fc
import solve
import triangulation as tr
import numpy as np

if __name__ == "__main__":
    t = tr.rectangle_triangulation(rec = [gm.point([0.0, 0.0]), gm.point([1.0, 1.0])],
                                   x_mesh_num = 10, y_mesh_num = 10)
    eq = solve.solve(tr = t, main_function = fc.Cos(1.0, 2.0, 0.0), boundary_function = fc.Sin(2.0, 3.0, 0.0))
    x = gm.point([0.5, 0.6])
    print(eq.eval(x))
