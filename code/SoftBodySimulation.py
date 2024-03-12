import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from SBSLoadVis import Mesh, Scene, load_scene_file, flatten_meshes, load_constraint_file, update_visual_constraints
import cppyy
import cppyy.gbl


def callback():
    # Executed every frame
    global CRCoeff, timeStep, isAnimating, scene, constraints

    # UI stuff
    psim.PushItemWidth(50)

    psim.TextUnformatted("Animation Parameters")
    psim.Separator()
    changed, isAnimating = psim.Checkbox("isAnimating", isAnimating)

    psim.PopItemWidth()

    # Actual animation
    if not isAnimating:
        return

    scene.run_timestep(constraints=constraints)

    currVertices, _, _,_ = flatten_meshes(scene.meshes, flattenFaces=False)
    constVertices, _ = update_visual_constraints(ps_mesh, currVertices, allConstEdges)
    ps_mesh.update_vertex_positions(currVertices)
    ps_constraints.update_node_positions(constVertices)
    # isAnimating = False


if __name__ == '__main__':
    ps.init()

    # initializing ccd code
    exec(open("vec3.py").read())
    exec(open("support.py").read())
    exec(open("mpr.py").read())
    exec(open("ccd.py").read())
    exec(open("isCollide.py").read())

    from cppyy.gbl import isCollide

    # dummy code to initialize libccd
    v1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    v2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) + 0.1
    COM1 = np.mean(v1, axis=0)
    COM2 = COM1 + 0.1
    n1 = np.array([4.0])
    n2 = np.array([4.0])
    depth = np.array([0.0])
    intNormal = np.array([0.0, 0.0, 0.0])
    intPosition = np.array([0.0, 0.0, 0.0])
    result = cppyy.gbl.isCollide(n1, v1, COM1, n2, v2, COM2, depth, intNormal, intPosition)

    timeStep = 0.02  # assuming 50 fps
    alpha = beta = 0.1
    # creating pseudo-mesh for the ground for the purpose of collision resolution
    groundMesh = Mesh(-1, np.empty([0, 3]), np.empty([0, 3]), np.empty([0, 4]),  0.0, np.zeros([1, 3]),
                      np.array([1.0, 0.0, 0.0, 0.0]), True, 1000.0, 1000, timeStep, alpha, beta)

    scene = load_scene_file(os.path.join('..', 'data', 'cylinder-scene.txt'), timeStep, groundMesh, alpha, beta)

    ###Change here for unlocking constraint enforcing
    enforceConstraints = True
    if (enforceConstraints):
        constraints, visConstraints = load_constraint_file(os.path.join('..', 'data', 'cylinder-constraints.txt'), scene)
    else:
        constraints = []
        visConstraints = []

    ####GUI stuff
    isAnimating = False
    allVertices, allFaces, allDensities, allConstEdges = flatten_meshes(scene.meshes, constraints=visConstraints)
    sceneBBox = (allVertices.min(axis=0), allVertices.max(axis=0))
    sceneCenter = allVertices.mean(axis=0)
    ps_mesh = ps.register_surface_mesh("Entire scene", allVertices, allFaces, smooth_shade=True)
    ps_mesh.add_scalar_quantity("density", allDensities)
    constVertices, constEdges = update_visual_constraints(ps_mesh, allVertices, allConstEdges)
    ps_constraints = ps.register_curve_network("Constraints", constVertices, constEdges)
    ps.set_bounding_box([sceneBBox[1][0], 0, sceneBBox[1][2]], sceneBBox[1])
    ps.set_ground_plane_height_factor(0.0, False)
    ps.set_user_callback(callback)

    ps.show()

    ps.clear_user_callback()
