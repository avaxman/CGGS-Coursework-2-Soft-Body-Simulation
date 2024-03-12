import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.join('..', 'code'))
import meshio
from SBSFunctions import Mesh

if __name__ == '__main__':

    data_path = os.path.join('..', 'data')  # Replace with the path to your folder
    mesh_files = [file for file in os.listdir(data_path) if file.endswith(".mesh")]
    timeStep = 0.02
    for currFileIndex in range(len(mesh_files)):
        print("Processing mesh ", mesh_files[currFileIndex])
        mesh_file_path = os.path.join(data_path, mesh_files[currFileIndex])
        tetMesh = meshio.read(mesh_file_path)
        faces = tetMesh.cells[0].data if tetMesh.cells[0].type == 'triangle' else tetMesh.cells[1].data
        tets = tetMesh.cells[1].data if tetMesh.cells[0].type == 'triangle' else tetMesh.cells[0].data
        vertices = tetMesh.points

        root, old_extension = os.path.splitext(mesh_file_path)
        pickle_file_path = root + '-FEM-matrices.data'

        with open(pickle_file_path, 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)

        mesh = Mesh(0, vertices, faces, tets, loaded_data['density'], np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), False,
                    loaded_data['youngModulus'],
                    loaded_data['poissonRatio'], timeStep, loaded_data['alpha'], loaded_data['beta'])

        print("mu error: ", np.max(np.abs(loaded_data['mu'] - mesh.mu)))
        print("lambda error: ", np.max(np.abs(loaded_data['lamb'] - mesh.lamb)))
        print("M error: ", np.max(np.abs(loaded_data['M'] - mesh.M)))
        print("K error: ", np.max(np.abs(loaded_data['K'] - mesh.K)))
        print("D error: ", np.max(np.abs(loaded_data['D'] - mesh.D)))