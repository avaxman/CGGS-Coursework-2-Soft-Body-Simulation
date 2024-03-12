import os
import numpy as np
import cppyy
import cppyy.gbl


def initialize_libccd():
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
