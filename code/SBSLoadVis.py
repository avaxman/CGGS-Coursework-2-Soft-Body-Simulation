import os
import numpy as np
from SBSFunctions import Scene, Mesh, Constraint, ConstraintType
import meshio
import cppyy
import scipy.linalg


def update_visual_constraints(ps_mesh, vertices, constEdges):
    constVertices = vertices[constEdges]

    constVertices = constVertices.reshape(2 * constVertices.shape[0], 3)
    curveNetIndices = np.arange(0, constVertices.shape[0])
    curveNetIndices = curveNetIndices.reshape(int(len(curveNetIndices) / 2), 2)
    return constVertices, curveNetIndices

    return constVertices, constEdges


def load_constraint_file(file_path, scene):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    numConstraints = int(lines[0].strip())
    constraints = []
    visConstraints = []
    for lineIndex in range(numConstraints):
        parts = lines[1 + lineIndex].split()
        fullConstraint = list(map(int, parts[0:4])) + list(map(float, parts[4:6]))

        visConstraints.append(fullConstraint[0:4])

        # computing original distance and putting it as the ref value for the constraints
        origDistance = np.linalg.norm(
            scene.meshes[fullConstraint[0]].currVertices[fullConstraint[1]] - scene.meshes[fullConstraint[2]].currVertices[
                fullConstraint[3]])

        vertexIndices = np.array((fullConstraint[1], fullConstraint[3])).astype(int)
        globalIndices = np.zeros(len(vertexIndices) * 3, dtype=int)
        globalIndices[::3] = 3 * vertexIndices
        globalIndices[1::3] = 3 * vertexIndices + 1
        globalIndices[2::3] = 3 * vertexIndices + 2
        globalIndices[0:3] += scene.meshes[fullConstraint[0]].globalOffset
        globalIndices[3:6] += scene.meshes[fullConstraint[2]].globalOffset
        invMasses = np.concatenate([scene.meshes[fullConstraint[0]].invMass * np.ones(3), scene.meshes[fullConstraint[2]].invMass * np.ones(3)])
        # lbConstraints = fullConstraint[0:4]+list((fullConstraint[4] * origDistance,False))
        # ubConstraints = fullConstraint[0:4]+list((fullConstraint[5] * origDistance, True))
        # constraints.append(lbConstraints)
        # constraints.append(ubConstraints)
        constraint = Constraint(ConstraintType.DISTANCE, globalIndices, invMasses, refValue = origDistance, meshIndices = np.array(
            (fullConstraint[0], fullConstraint[2]), dtype=int))
        constraints.append(constraint)
    return constraints, visConstraints


def load_scene_file(file_path, timeStep, groundMesh, alpha, beta):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the MESH file
    numMeshes = int(lines[0].strip())
    meshes = []
    for lineIndex in range(numMeshes):
        elements = lines[1 + lineIndex].split()
        directory, filename = os.path.split(file_path)

        mesh_file_path = os.path.join(directory, elements[0])
        # origVertices, faces, tets = load_mesh_file(mesh_file_path)
        mesh = meshio.read(mesh_file_path)
        boundF = mesh.cells[0].data if mesh.cells[0].type == 'triangle' else mesh.cells[1].data
        tets = mesh.cells[1].data if mesh.cells[0].type == 'triangle' else mesh.cells[0].data
        meshes.append(
            Mesh(lineIndex, mesh.points, boundF, tets, np.array(elements[1], dtype=float),
                 np.array(elements[5:8], dtype=float).reshape(1, 3),
                 np.array(elements[8:12], dtype=float), bool(int(elements[4])), float(elements[2]), float(elements[3]), timeStep, alpha, beta))

    scene = Scene(meshes, groundMesh, timeStep)

    return scene


def flatten_meshes(meshes, flattenFaces=True, constraints=[]):
    allVertices = np.empty((0, 3))
    allFaces = np.empty((0, 3))
    allDensities = np.empty((0))
    allConstEdges = np.zeros([len(constraints), 2], dtype=int)
    faceOffsets = np.zeros(len(meshes))
    faceOffsets[0] = 0
    currFaceOffset = 0
    currIndex = 0
    for mesh in meshes:
        allVertices = np.concatenate((allVertices, mesh.currVertices))
        if flattenFaces:
            allFaces = np.concatenate((allFaces, mesh.boundFaces + currFaceOffset))
            allDensities = np.concatenate((allDensities, np.full(mesh.currVertices.shape[0], mesh.density)))
            faceOffsets[currIndex] = currFaceOffset
            currFaceOffset += mesh.currVertices.shape[0]
            currIndex += 1

    # Translating constraints
    if (flattenFaces):
        for constraintIndex in range(len(constraints)):
            allConstEdges[constraintIndex, :] = [
                faceOffsets[constraints[constraintIndex][0]] + constraints[constraintIndex][1],
                faceOffsets[constraints[constraintIndex][2]] + constraints[constraintIndex][3]]

    return allVertices, allFaces, allDensities, allConstEdges


