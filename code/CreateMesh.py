import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import tetgen
import meshio


def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces


if __name__ == '__main__':
    ps.init()

    vertices, faces = load_off_file(os.path.join('..', 'data', 'cube_tri_refined.off'))

    tet = tetgen.TetGen(vertices, faces)
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)

    cells = [
        ("triangle", tet.f),
        ("tetra", tet.elem),
    ]

    mesh = meshio.Mesh(tet.node, cells)
    mesh.write(os.path.join('..', 'data', 'cube_tri_refined.mesh'))

    ####GUI stuff
    ps_mesh = ps.register_surface_mesh("OFF Mesh", vertices, faces, smooth_shade=True)
    ps.register_surface_mesh("Tetgen Boundary Mesh", tet.v, tet.f, smooth_shade=True)
    ps.register_volume_mesh("Tetgen Tet Mesh", tet.node, tets=tet.elem)

    ps.show()

    ps.clear_user_callback()
