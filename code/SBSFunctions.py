import os
import numpy as np
import cppyy
import scipy.linalg
from Quaternion import *
from scipy.sparse import csr_matrix, vstack, hstack, dia_matrix, coo_matrix, diags
from enum import Enum


def accumarray(indices, values):
    output = np.zeros((np.max(indices) + 1), dtype=values.dtype)
    indFlat = indices.flatten()
    valFlat = values.flatten()
    # for index in range(indFlat.shape[0]):
    #     output[indFlat[index]] += valFlat[index]
    np.add.at(output, indFlat, valFlat)

    return output


def single2coord(arr):
    # Reshape the input array to a column vector
    arr_column = arr[:, np.newaxis]

    # Create an array of consecutive numbers from 0 to 2
    consecutive_nums = np.arange(3)

    # Multiply the consecutive numbers by 3 and add to the reshaped input array
    result_array = 3 * arr_column + consecutive_nums

    # Flatten the result array to get a 1D array
    return result_array.flatten()


class ConstraintType(Enum):
    DISTANCE = 1
    COLLISION = 2


class Constraint:
    def __init__(self, constType, globalIndices, invMasses, refVector=np.zeros((0, 3)), refValue=0.0, meshIndices=np.array((-1,-1), dtype=int)):
        self.constType = constType
        self.globalIndices = globalIndices
        self.refVector = refVector
        self.refValue = refValue
        self.invMassMatrix = np.diag(invMasses)
        self.meshIndices = meshIndices

    def get_value_gradient(self, currPositions):
        if self.constType == ConstraintType.DISTANCE:
            fullPoint1 = currPositions[0:3]
            fullPoint2 = currPositions[3:6]
            posDiff = fullPoint1 - fullPoint2
            distance = np.linalg.norm(posDiff)
            value = distance - self.refValue
            gradient = np.hstack((posDiff / distance, -posDiff / distance))

        if self.constType == ConstraintType.COLLISION:
            value = np.dot(self.refVector, currPositions)
            gradient = self.refVector

        return value, gradient

    def resolve_positions(self, currPositions, tolerance):

        value, gradient = self.get_value_gradient(currPositions)
        if (self.constType == ConstraintType.COLLISION) & (value > -tolerance):
            posCorrection = np.zeros(len(self.globalIndices))
            return True, []

        if (self.constType == ConstraintType.DISTANCE) & (np.abs(value) < tolerance):
            posCorrection = np.zeros(len(self.globalIndices))
            return True, []

        # otherwise resolve the constraint
        denominator = gradient @ self.invMassMatrix @ gradient.transpose()
        lagMult = -value / denominator
        posCorrections = (self.invMassMatrix @ gradient.transpose()) * lagMult

        return False, posCorrections


class Scene:
    def __init__(self, meshes, groundMesh, timeStep):
        self.meshes = meshes
        self.groundMesh = groundMesh
        self.timeStep = timeStep
        self.M = []
        self.K = []
        self.D = []
        self.globalOffsets = np.zeros(len(self.meshes))

        self.globalOrigPositions = np.zeros(0)
        self.globalCurrPositions = np.zeros(0)
        self.globalVelocities = np.zeros(0)
        self.localConstraints = []

        # computing global indices
        self.totalNumVertices = 0
        currMeshIndex = 0
        for mesh in self.meshes:
            mesh.globalOffset = self.totalNumVertices * 3
            self.globalOrigPositions = np.concatenate((self.globalOrigPositions, mesh.origVertices.flatten()))
            self.globalOffsets[currMeshIndex] = mesh.globalOffset
            self.totalNumVertices += mesh.origVertices.shape[0]
            currMeshIndex += 1

        self.globalCurrPositions = np.copy(self.globalOrigPositions)
        self.globalVelocities = np.zeros_like(self.globalOrigPositions)
        self.compute_global_matrices()

    def global2mesh(self):
        # TODO: copy global values to mesh
        for meshIndex in np.arange(len(self.meshes)- 1):
            self.meshes[meshIndex].currVertices = self.globalCurrPositions[
                                                  self.meshes[meshIndex].globalOffset:self.meshes[
                                                      meshIndex + 1].globalOffset]
            self.meshes[meshIndex].velocities = self.globalVelocities[
                                                self.meshes[meshIndex].globalOffset:self.meshes[
                                                    meshIndex + 1].globalOffset]

        self.meshes[-1].currVertices = self.globalCurrPositions[self.meshes[-1].globalOffset:]
        self.meshes[-1].velocities = self.globalVelocities[self.meshes[-1].globalOffset:]

        for mesh in self.meshes:
            mesh.currVertices = np.reshape(mesh.currVertices, (mesh.origVertices.shape[0], 3))
            mesh.velocities = np.reshape(mesh.velocities, (mesh.origVertices.shape[0], 3))

    def compute_global_matrices(self):

        #TODO: aggregate all local mesh matrices to global **sparse** ones, and form the left-hand side matrix for velocity integration

        #stubs
        self.M = []
        self.K = []
        self.D = []
        self.A = []

    def run_timestep(self, constraints=None, maxIterations=10000, tolerance=1e-3):

        # Constrained integration of velocities and positions
        self.integrate_timestep()
        # print("min y of scene after: ", np.min(self.globalCurrPositions[1::3]))

        self.global2mesh()  # copying back values to individual meshes

        # first figuring out collisions and resolving the positions
        self.localConstraints = constraints.copy()
        for mesh1 in self.meshes:
            for mesh2 in self.meshes[self.meshes.index(mesh1) + 1:]:  # TODO: use range
                self.localConstraints += mesh1.detect_collision(mesh2)

        # Creating collision constraints for the ground
        for meshIndex in np.arange(len(self.meshes)):
            mesh = self.meshes[meshIndex]
            minyIndex = int(np.argmin(mesh.currVertices, axis=0)[1])
            if mesh.currVertices[minyIndex, 1] <= 0.0:
                groundCollConstraint = Constraint(ConstraintType.COLLISION, mesh.globalOffset + np.array((
                    3 * minyIndex, 3 * minyIndex + 1, 3 * minyIndex + 2), dtype = int), np.array((mesh.invMass, mesh.invMass, mesh.invMass)),
                                                  refVector=np.array([0.0, 1.0, 0.0]).reshape(1, 3), meshIndices = np.array((meshIndex, -1), dtype = int))
                self.localConstraints.append(groundCollConstraint)

                # resolving position immediately
                startPos = mesh.globalOffset
                endPos = self.meshes[meshIndex + 1].globalOffset if meshIndex != len(self.meshes)-1 else None
                #updating ust the y coordinate
                self.globalCurrPositions[startPos+1:endPos:3] -= mesh.currVertices[minyIndex, 1]

        # print("min y of scene before: ", np.min(self.globalCurrPositions[1::3]))

        # Resolving position constraints
        self.resolve_all_constraint_positions(maxIterations, tolerance)

        self.global2mesh()  # copying back values to individual meshes


    def resolve_all_constraint_positions(self, maxIterations, tolerance):
        # sequentially resolving position constraints
        currIteration = 0
        zeroStreak = 0
        currConstIndex = 0
        while zeroStreak < len(self.localConstraints) and currIteration * len(self.localConstraints) < maxIterations:
            currConstraint = self.localConstraints[currConstIndex]
            if len(currConstraint.globalIndices) == 3:  # a collision with the ground mesh
                zeroStreak += 1
                continue  # already been dealt with in the detection phase
            currMeshIndices = currConstraint.meshIndices
            currPositions = self.globalCurrPositions[currConstraint.globalIndices]
            positionWasValid, posCorrections = currConstraint.resolve_positions(currPositions, tolerance)
            if not positionWasValid:
                posCorrections = np.reshape(posCorrections, (len(posCorrections)//3, 3))
                # finding the single pos correction for the point and taking this to the whole mesh
                correctPosMesh1 = np.mean(posCorrections[0:posCorrections.shape[0] // 2, :],
                                          axis=0)  # CHECk THIS IS CORRECT!
                correctPosMesh2 = np.mean(posCorrections[posCorrections.shape[0] // 2:posCorrections.shape[0], :],
                                          axis=0)

                meshIndex1 = currMeshIndices[0]
                meshIndex2 = currMeshIndices[1]

                startPos1 = self.meshes[meshIndex1].globalOffset
                endPos1 = self.meshes[meshIndex1 + 1].globalOffset if meshIndex1 != len(self.meshes)-1 else None
                self.globalCurrPositions[startPos1:endPos1] += np.tile(correctPosMesh1, self.meshes[meshIndex1].origVertices.shape[0])

                startPos2 = self.meshes[meshIndex2].globalOffset
                endPos2 = self.meshes[meshIndex2 + 1].globalOffset if meshIndex2 != len(self.meshes)-1 else None
                self.globalCurrPositions[startPos2:endPos2] += np.tile(correctPosMesh2, self.meshes[meshIndex2].origVertices.shape[0])

                zeroStreak = 0
            else:
                zeroStreak += 1

            currIteration += 1
            currConstIndex = (currConstIndex + 1) % (len(self.localConstraints))


    def integrate_global_velocity(self):
        ##TODO: the local integration step  for velocities

        #stub
        self.globalVelocities = []

    def integrate_global_position(self):

        #TODO: integration of positions
        self.globalCurrPositions = []

    def integrate_timestep(self):
        # semi-implicit Euler integration
        self.integrate_global_velocity()
        self.integrate_global_position()


    def compute_all_constraints_jacobian(self, constraints):
        currRow = 0
        JRows = np.zeros(0)
        JCols = np.zeros(0)
        JValues = np.zeros(0)
        for constraint in constraints:
            JRows = np.concatenate((JRows, np.full(len(constraint.globalIndices), currRow, dtype=int)))
            currRow += 1
            JCols = np.concatenate((JCols, constraint.globalIndices))
            _, localJValues = constraint.get_value_gradient(self.globalCurrPositions[constraint.globalIndices])
            JValues = np.concatenate((JValues, localJValues.squeeze()))

        J = coo_matrix((JValues, (JRows, JCols)), shape = (len(constraints), len(self.globalCurrPositions)))
        return J


class Mesh:

    def __init__(self,  index, origVertices, boundFaces, tets, density, position, orientation, isFixed, youngModulus,
                 poissonRatio, timeStep, alpha, beta):
        # Instance attributes
        self.index = index
        self.origVertices = np.copy(origVertices)
        self.velocities = np.zeros_like(self.origVertices)
        self.density = density
        self.boundFaces = boundFaces
        self.tets = tets
        self.isFixed = isFixed
        self.timeStep = timeStep
        self.youngModulus = youngModulus
        self.poissonRatio = poissonRatio
        self.alpha = alpha
        self.beta = beta
        self.globalOffset = -1

        if self.origVertices.shape[0] > 0:
            self.init_static_properties()
            # initial orientation
            self.origVertices = QRotate(orientation.reshape(1, 4), self.origVertices) + position
            self.currVertices = np.copy(self.origVertices)
            self.invMass = 1.0 / self.mass

        if self.origVertices.shape[0] == 0 or self.isFixed:
            self.invMass = 0.0

        # finding boundary tets
        if self.origVertices.shape[0]!=0:
            boundVMask = np.zeros(origVertices.shape[0])
            boundVMask[np.unique(self.boundFaces.flatten())] = 1
            boundTIncidence = np.sum(boundVMask[self.tets], axis=1)
            self.boundTets = list(np.where(boundTIncidence > 2)[0])
            self.compute_soft_body_matrices()

    def compute_soft_body_matrices(self):
        ####TODO: compute the Lame parameters, and the FEM soft-body matrices
        #stubs
        self.mu = 0.0
        self.lamb = 0.0
        self.M = []
        self.K = []
        self.D = []


    def init_static_properties(self):
        # obtaining the natural COM of the original vertices an putting it to (0,0,0) so it's easier later
        e01 = self.origVertices[self.tets[:, 1], :] - self.origVertices[self.tets[:, 0], :]
        e02 = self.origVertices[self.tets[:, 2], :] - self.origVertices[self.tets[:, 0], :]
        e03 = self.origVertices[self.tets[:, 3], :] - self.origVertices[self.tets[:, 0], :]
        tetCentroids = (self.origVertices[self.tets[:, 0], :] + self.origVertices[self.tets[:, 1],
                                                                :] + self.origVertices[self.tets[:, 2],
                                                                     :] + self.origVertices[self.tets[:, 3], :]) / 4.0
        self.tetVolumes = np.abs(np.sum(e01 * np.cross(e02, e03), axis=1)) / 6.0
        totalVolume = np.sum(self.tetVolumes)

        naturalCOM = np.sum(tetCentroids * self.tetVolumes[:, np.newaxis], axis=0) / totalVolume
        self.origVertices -= naturalCOM

        self.mass = self.density * totalVolume
        self.tetMasses = self.density * self.tetVolumes
        self.vertexMasses = accumarray(self.tets.flatten(), np.repeat(self.tetMasses.reshape((len(self.tetMasses), 1)), 4, axis = 1).flatten()) / 4.0

    # Return true if all dimensions are overlapping
    def bounding_box_collision(self, m):
        selfBBox = [np.min(self.currVertices, axis=0), np.max(self.currVertices, axis=0)]
        mBBox = [np.min(m.currVertices, axis=0), np.max(m.currVertices, axis=0)]
        return np.all(selfBBox[1] >= mBBox[0]) and np.all(mBBox[1] >= selfBBox[0])

    def tet_bounding_box_collision(self, v1, v2):
        selfBBox = [np.min(v1, axis=0), np.max(v1, axis=0)]
        mBBox = [np.min(v2, axis=0), np.max(v2, axis=0)]
        return np.all(selfBBox[1] >= mBBox[0]) and np.all(mBBox[1] >= selfBBox[0])

    def detect_collision(self, m):
        if self.isFixed and m.isFixed:
            return []

        if not self.bounding_box_collision(m):
            return []

        possibleTetCollisions = []
        # Checking individual bounding box collisions between tets
        tetPairs = ((x, y) for x in self.boundTets for y in m.boundTets)
        for currPair in tetPairs:
            if not self.tet_bounding_box_collision(self.currVertices[self.tets[currPair[0], :], :], m.currVertices[m.tets[currPair[1], :], :]):
                continue
            else:  # aggregate collision to check later
                possibleTetCollisions.append(currPair)

        collisionList = []
        for currPossibleTetCollision in possibleTetCollisions:
            # performing full CCD algorithm

            n1 = np.array([float(4)])
            n2 = np.array([float(4)])

            # Initialization to be able to pass pointers
            depth = np.array([0.0])
            intNormal = np.array([0.0, 0.0, 0.0])
            intPosition = np.array([0.0, 0.0, 0.0])

            tet1 = self.tets[currPossibleTetCollision[0], :]
            tet2 = m.tets[currPossibleTetCollision[1], :]
            #TODO: fix according to 3
            vertexIndices = np.concatenate((tet1, tet2)).astype(int)
            globalIndices = np.zeros(len(vertexIndices)*3, dtype=int)
            globalIndices[::3] = 3*vertexIndices
            globalIndices[1::3] = 3*vertexIndices + 1
            globalIndices[2::3] = 3*vertexIndices + 2
            globalIndices[0:12] += self.globalOffset
            globalIndices[12:24] += m.globalOffset
            invMasses = np.concatenate([self.invMass * np.ones(12), m.invMass * np.ones(12)])
            # Giving the full masses of the objects, as we resolve positions by moving the entire object

            # Call the C++ function with the Eigen matrices and ctypes for double by reference
            selfTetCentre = np.mean(self.currVertices[tet1, :], axis=0)
            mTetCentre = np.mean(m.currVertices[tet2, :], axis=0)
            isCollision = cppyy.gbl.isCollide(n1, self.currVertices[tet1, :],
                                              selfTetCentre, n2,
                                              m.currVertices[tet2, :], mTetCentre,
                                              depth,
                                              intNormal, intPosition)

            if not isCollision:
                continue

            # creating constraint of two tets with barycentric vectors
            p1 = intPosition + depth * intNormal
            p2 = intPosition

            tetVertices1 = self.currVertices[tet1, :]
            PMat1 = np.hstack((np.ones((4,1)), tetVertices1))
            PMat1 = PMat1.T
            rhs1 = np.concatenate(([1.0], p1.squeeze()))
            B1 = np.linalg.inv(PMat1) @ rhs1

            tetVertices2 = m.currVertices[tet2, :]
            PMat2 = np.hstack((np.ones((4,1)), tetVertices2))
            PMat2 = PMat2.T

            rhs2 = np.concatenate(([1.0], p2.squeeze()))
            B2 = np.linalg.inv(PMat2) @ rhs2

            v2cMat1 = np.hstack((np.eye(3) * B1[0], np.eye(3) * B1[1], np.eye(3) * B1[2], np.eye(3) * B1[3]))
            v2cMat2 = np.hstack((np.eye(3) * B2[0], np.eye(3) * B2[1], np.eye(3) * B2[2], np.eye(3) * B2[3]))

            v2dMat = np.hstack([-v2cMat1, v2cMat2])
            constVector = intNormal.T @ v2dMat

            colConstraint = Constraint(ConstraintType.COLLISION, globalIndices, invMasses, constVector, refValue=0.0, meshIndices = np.array((self.index, m.index),dtype = int))
            collisionList.append(colConstraint)

        return collisionList
