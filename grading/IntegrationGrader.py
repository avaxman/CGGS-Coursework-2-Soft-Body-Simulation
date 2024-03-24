import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.join('..', 'code'))
from SBSFunctions import Scene, Mesh
from SBSLoadVis import load_scene_file, load_constraint_file
import meshio


if __name__ == '__main__':

    data_path = os.path.join('..', 'data')  # Replace with the path to your folder
    timeStep = 0.02
    alpha = beta = 0.1
    pickle_file_path = data_path + os.path.sep + 'integration.data'

    with open(pickle_file_path, 'rb') as pickle_file:
        loaded_data = pickle.load(pickle_file)

    groundMesh = Mesh(-1, np.empty([0, 3]), np.empty([0, 3]), np.empty([0, 4]), 0.0, np.zeros([1, 3]),
                      np.array([1.0, 0.0, 0.0, 0.0]), True, 1000.0, 1000, timeStep, alpha, beta)
    # first integrating without constraints
    scene_file_path = data_path + os.path.sep + 'cylinder-scene.txt'
    scene = load_scene_file(scene_file_path, timeStep, groundMesh, alpha, beta)

    # artificially compressing one mesh and blowing up another
    scene.globalCurrPositions[0:scene.meshes[1].globalOffset:3] *= 1.3
    scene.globalCurrPositions[scene.meshes[1].globalOffset + 2::3] *= 0.7

    scene.integrate_global_velocity()
    scene.integrate_global_position()

    globalVelocitiesFree = np.copy(scene.globalVelocities)
    globalPositionsFree = np.copy(scene.globalCurrPositions)

    constraints, _ = load_constraint_file(os.path.join('..', 'data', 'cylinder-constraints.txt'), scene)
    scene.localConstraints = constraints

    scene.run_timestep(constraints)

    globalVelocitiesConstrained = np.copy(scene.globalVelocities)
    globalPositionsConstrained = np.copy(scene.globalCurrPositions)

    print("Free global velocities error: ", np.max(np.abs(globalVelocitiesFree - loaded_data['globalVelocitiesFree'])))
    print("Free global positions error: ", np.max(np.abs(globalPositionsFree - loaded_data['globalPositionsFree'])))
    print("Constrained global velocities error: ", np.max(np.abs(globalVelocitiesConstrained - loaded_data['globalVelocitiesConstrained'])))
    print("Free global positions error: ", np.max(np.abs(globalPositionsConstrained - loaded_data['globalPositionsConstrained'])))

