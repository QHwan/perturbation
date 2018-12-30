import numpy as np
cimport numpy as np

cdef vec_minus(float *ijVec, float *iVec, float *jVec)
cdef vec_dot(float *iVec, float *jVec)

cdef integral_range(float Din, float loutn, float linn, float linp, float loutp)
cdef cal_force(float *fVec, float **iMat, float **jMat)

cdef cal_pn(float *fVec, float *iVec, float *jVec, la, lb)
cdef cal_pt(float *fVec, float *iVec, float *jVec, la, lb)

cdef coord_pert(float **iMat, float **bufMat, float epsp)
cdef coord_pert_area(float **iMat, float **bufMat, float epsp)

cdef cal_LJ(float **iMat, float **jMat)
cdef cal_C(float **iMat, float **jMat, float *ijVec)


cdef cal_ptensor(np.ndarray[float,ndim=1] rVec, np.ndarray[float,ndim=1] pkVec, np.ndarray[float,ndim=1] p1Vec, np.ndarray[float,ndim=1] ptVec, np.ndarray[float,ndim=1] pnVec)

cdef cal_tension(np.ndarray[float,ndim=1] rVec, np.ndarray[float,ndim=1] ptVec, np.ndarray[float,ndim=1] pnVec)

