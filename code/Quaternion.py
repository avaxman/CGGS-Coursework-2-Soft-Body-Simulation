import numpy as np
import scipy.linalg


def QMult(q1, q2):
    # r = (q1[:, 0] * q2[:, 0] - np.sum(q1[:, 1:4] * q2[:, 1:4], axis=1)).reshape(q1.shape[0], 1)
    # v = q1[:, 0].reshape(q1.shape[0], 1) * q2[:, 1:4] + q2[:, 0].reshape(q2.shape[0], 1) * q1[:, 1:4] + np.cross(
    #    q1[:, 1:4], q2[:, 1:4])
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.column_stack((w, x, y, z))

def QConj(q):
    return np.hstack((q[:, 0].reshape(q.shape[0], 1), -q[:, 1:4].reshape(q.shape[0], 3)))


def QInv(q):
    qabs = np.linalg.norm(q, axis=1).reshape(q.shape[0], 1)
    return QConj(q) / (qabs ** 2)


def QRotate(q1, v):
    if (q1.shape[0] == 1):  # broadcasting a single quaternion
        q1r = np.tile(q1, (v.shape[0], 1))
    else:
        q1r = q1
    return QMult(QMult(q1r, np.hstack((np.zeros((v.shape[0], 1)), v))), QInv(q1r))[:, 1:4]


def QExp(q):
    absv = np.linalg.norm(q[:, 1:4], axis=1).reshape(q.shape[0], 1)
    if np.abs(absv) < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    else:
        eq = np.exp(q[:, 0]) * np.hstack((np.cos(absv).reshape(absv.shape[0], 1), np.sin(absv) * q[:, 1:4] / absv))
        # print("eq norm", np.linalg.norm(eq))
        return eq


def Q2RotMatrix(q):
    w, x, y, z = q[0]
    R = np.array([
        [1 - 2*(y**2) - 2*(z**2), 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*(x**2) - 2*(z**2), 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*(x**2) - 2*(y**2)]
    ])
    return R
