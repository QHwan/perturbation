import numpy as np
cimport numpy as np
import math
import scipy.integrate
from libc.math cimport sqrt, log, atan
from libc.stdlib cimport malloc, free

cdef vec_minus(float *ijVec, float *iVec, float *jVec):
	cdef int i

	for i in range (3):
		ijVec[i] = jVec[i] - iVec[i]


cdef vec_dot(float *iVec, float *jVec):
	cdef int i
	cdef float d

	d = 0
	for i in range (3):
		d += iVec[i] * jVec[i]
	
	return d


cdef integral_range(float Din, float loutn, float linn, float linp, float loutp):
	cdef float la, lb, lap, lbp

	if linn<0 and loutn<0:
		la = linp
		lb = loutp
		lap = -10
		lbp = -10
		if linp<0:
			la = 0
		if loutp>1:
			lb = 1
	elif linp>1 and loutp>1:
		la = loutn
		lb = linn
		lap = -10
		lbp = -10
		if loutn<0:
			la = 0
		if linn>1:
			lb = 1
	elif Din<0:
		la = loutn
		lb = loutp
		lap = -10
		lbp = -10
		if loutn<0:
			la = 0
		if loutp>1:
			lb = 1
	else:
		la = loutn
		lb = linn
		lap = linp
		lbp = loutp
		if loutn<0:
			la = 0
		if loutp>1:
			lb = 1
	return la, lb, lap, lbp


cdef cal_LJ_force(float *ijVec, float rij, float sig6, float eps):
	cdef float f
	cdef float fx, fy, fz
	cdef np.ndarray[float, ndim=1] fLJVec

	if sig6==0 or eps==0:
		fLJVec = np.zeros(3, dtype=np.float32)
	else:

		f = 12*sig6*sig6/pow(rij,13) - 6*sig6/pow(rij,7)
		f *= 4*eps

		fLJVec = np.zeros(3, dtype=np.float32)
		fLJVec[0] = f*ijVec[0]/rij
		fLJVec[1] = f*ijVec[1]/rij
		fLJVec[2] = f*ijVec[2]/rij

	return fLJVec


cdef cal_C_force(float *ijVec, float rij, float qi, float qj):
	cdef float f
	cdef float pref
	cdef float fx, fy, fz
	cdef np.ndarray[float, ndim=1] fCVec

	if qi==0 or qj==0:
		fCVec = np.zeros(3, dtype=np.float32)
	else:
		pref = 138.935458

		f = pref*qi*qj/rij/rij/rij

		fCVec = np.zeros(3, dtype=np.float32)
		fCVec[0] = f*ijVec[0]
		fCVec[1] = f*ijVec[1]
		fCVec[2] = f*ijVec[2]

	return fCVec


cdef cal_force(float *fVec, float **iMat, float **jMat):
	cdef int i, j
	cdef int apm
	cdef float sig, eps, sig6
	cdef float qh, qm
	cdef float r
	cdef float rij
	cdef float f, pref
	cdef np.ndarray[float, ndim=1] siVec
	cdef np.ndarray[float, ndim=1] sjVec
	cdef np.ndarray[float, ndim=1] eiVec
	cdef np.ndarray[float, ndim=1] ejVec
	cdef np.ndarray[float, ndim=1] qiVec
	cdef np.ndarray[float, ndim=1] qjVec
	cdef np.ndarray[float, ndim=1] bufVec
	cdef np.ndarray[float, ndim=1] fCVec
	cdef np.ndarray[float, ndim=1] fLJVec

	cdef float *ijVec

	ijVec = <float *> malloc(3 * sizeof(float))

	fLJVec = np.zeros(3, dtype=np.float32)
	fCVec = np.zeros(3, dtype=np.float32)



	sig = 0.31589
	sig6 = pow(sig,6)
	eps = 0.7749
	qh = 0.5564
	qm = -1.1128

	apm = 4

	siVec = np.zeros(apm, dtype=np.float32)
	sjVec = np.zeros(apm, dtype=np.float32)
	eiVec = np.zeros(apm, dtype=np.float32)
	ejVec = np.zeros(apm, dtype=np.float32)
	qiVec = np.zeros(apm, dtype=np.float32)
	qjVec = np.zeros(apm, dtype=np.float32)


	siVec[0] = sig6
	sjVec[0] = sig6
	eiVec[0] = eps
	ejVec[0] = eps
	qiVec[1] = qh
	qiVec[2] = qh
	qiVec[3] = qm
	qjVec[1] = qh
	qjVec[2] = qh
	qjVec[3] = qm


	for i in range (apm):
		for j in range (apm):
			for k in range (3):
				ijVec[k] = jMat[j][k] - iMat[i][k]
			rij = sqrt(vec_dot(ijVec,ijVec))

			if i==0 and j==0:
				f = 12*sig6*sig6/pow(rij,13) - 6*sig6/pow(rij,7)
				f *= 4*eps

				fLJVec[0] = f*ijVec[0]/rij
				fLJVec[1] = f*ijVec[1]/rij
				fLJVec[2] = f*ijVec[2]/rij

				#bufVec = cal_LJ_force(ijVec, rij, siVec[i], eiVec[j])
				for k in range (3):
					fVec[k] += fLJVec[k]

			if i!=0 and j!=0:
				pref = 138.935458

				f = pref*qiVec[i]*qjVec[j]/rij/rij/rij

				fCVec[0] = f*ijVec[0]
				fCVec[1] = f*ijVec[1]
				fCVec[2] = f*ijVec[2]

#				bufVec = cal_C_force(ijVec, rij, qiVec[i], qjVec[j])
				for k in range (3):
					fVec[k] += fCVec[k]

	free(ijVec)






cdef pn_core(a, b, c, d, e, f, l):
	pn = 2*c*f*l
	pn += (b*f - c*e)*log(d + e*l + f*l*l)
	pn += 2*atan( (e+2*f*l)/sqrt(4*d*f-e*e) )/sqrt(4*d*f - e*e) * (f*(2*a*f-b*e) + c*(e*e-2*d*f))
	pn *= 0.5/(f*f)
	return pn

cdef cal_pn(float *fVec, float *iVec, float *ijVec, la, lb):
	cdef int i
	cdef float a, b, c, d, e, f
	cdef float rifij, rijfij, ririj, riri, rijrij
	cdef float pn, pnb, pna

	rifij = 0.
	rijfij = 0.
	ririj = 0.
	riri = 0.
	rijrij = 0.
	for i in range (3):
		rifij += iVec[i]*fVec[i]
		rijfij += ijVec[i]*fVec[i]
		ririj += iVec[i]*ijVec[i]
		rijrij += ijVec[i]*ijVec[i]
		riri += iVec[i]*iVec[i]

	a = rifij * ririj
	b = rijfij * ririj + rifij * rijrij
	c = rijfij * rijrij
	d = riri
	e = 2*ririj
	f = rijrij

	pnb = 2*c*f*lb
	pnb += (b*f - c*e)*log(d + e*lb + f*lb*lb)
	pnb += 2*atan( (e+2*f*lb)/sqrt(4*d*f-e*e) )/sqrt(4*d*f - e*e) * (f*(2*a*f-b*e) + c*(e*e-2*d*f))
	pnb *= 0.5/(f*f)

	pna = 2*c*f*la
	pna += (b*f - c*e)*log(d + e*la + f*la*la)
	pna += 2*atan( (e+2*f*la)/sqrt(4*d*f-e*e) )/sqrt(4*d*f - e*e) * (f*(2*a*f-b*e) + c*(e*e-2*d*f))
	pna *= 0.5/(f*f)

	pn = pnb - pna

	return pn


cdef cal_pt(float *fVec, float *iVec, float *ijVec, la, lb):
	cdef float a, b, c, d, e, f
	cdef float pt, ptb, pta

	a = iVec[0]*fVec[1] - iVec[1]*fVec[0]
	b = ijVec[0]*fVec[1] - ijVec[1]*fVec[0]
	c = iVec[0]*ijVec[1] - iVec[1]*ijVec[0]
	d = iVec[0]*iVec[0] + iVec[1]*iVec[1]
	e = 2*(iVec[0]*ijVec[0] + iVec[1]*ijVec[1])
	f = ijVec[0]*ijVec[0] + ijVec[1]*ijVec[1]

	if 4*d*f-e*e < 0:
		pt = 0.
	else:
		
		ptb = b*log(d + e*lb + f*lb*lb)/(2*f)
		ptb += atan( (e+2*f*lb)/sqrt(4*d*f-e*e) )*(2*a*f - b*e)/f/sqrt(4*d*f - e*e) 
		ptb *= c

		pta = b*log(d + e*la + f*la*la)/(2*f)
		pta += atan( (e+2*f*la)/sqrt(4*d*f-e*e) )*(2*a*f - b*e)/f/sqrt(4*d*f - e*e) 
		pta *= c

		pt = ptb - pta

	return pt


cdef coord_pert(float **iMat, float **bufMat, float epsp):
	cdef int i, j
	cdef int apm

	apm = 4
	
	for i in range (1, apm):
		for j in range (3):
			bufMat[i][j] = iMat[i][j] - iMat[0][j]

	iMat[0][0] *= 1+epsp
	iMat[0][1] *= 1+epsp
	iMat[0][2] *= 1+epsp

	for i in range (1, apm):
		for j in range (3):
			iMat[i][j] = iMat[0][j] + bufMat[i][j]


cdef coord_pert_area(float **iMat, float **bufMat, float epsp):
	cdef int i, j
	cdef int apm

	apm = 4
	
	for i in range (1, apm):
		for j in range (3):
			bufMat[i][j] = iMat[i][j] - iMat[0][j]

	iMat[0][0] *= sqrt(1+epsp)
	iMat[0][1] *= sqrt(1+epsp)
	iMat[0][2] *= (1+epsp)**-1

	for i in range (1, apm):
		for j in range (3):
			iMat[i][j] = iMat[0][j] + bufMat[i][j]

cdef cal_LJ(float **iMat, float **jMat):
	cdef int i, j
	cdef int apm
	cdef float sig, eps, sig6
	cdef float rij
	cdef float pot
	
	sig = 0.31589
	sig6 = pow(sig,6)
	eps = 0.7749

	apm = 4

	rij = sqrt( (jMat[0][0]-iMat[0][0])**2 + (jMat[0][1]-iMat[0][1])**2 + (jMat[0][2]-iMat[0][2])**2 )

	pot = sig6*sig6/pow(rij,12) - sig6/pow(rij,6)
	
	pot *= 4*eps

	return pot


cdef cal_C(float **iMat, float **jMat, float *ijVec):
	cdef int i, j, k
	cdef int apm
	cdef float qh, qm
	cdef float r
	cdef float rij
	cdef float f, pref
	cdef float pot
	cdef np.ndarray[float, ndim=1] qVec



	qh = 0.5564
	qm = -1.1128

	apm = 4

	qVec = np.zeros(apm, dtype=np.float32)

	qVec[1] = qh
	qVec[2] = qh
	qVec[3] = qm

	pref = 138.935458

	pot = 0.
	for i in range (1, apm):
		for j in range (1, apm):
			for k in range (3):
				ijVec[k] = jMat[j][k] - iMat[i][k]
			rij = sqrt(ijVec[0]*ijVec[0] + ijVec[1]*ijVec[1] + ijVec[2]*ijVec[2])

			pot += qVec[i]*qVec[j]/rij
	pot *= pref



	return pot


cdef cal_ptensor(np.ndarray[float,ndim=1] rVec, np.ndarray[float,ndim=1] pkVec, np.ndarray[float,ndim=1] p1Vec, np.ndarray[float,ndim=1] ptVec, np.ndarray[float,ndim=1] pnVec):
	cdef int i, j
	cdef np.ndarray[float,ndim=1] xVec
	cdef np.ndarray[float,ndim=1] yVec
	for i in range (1,len(rVec)):
		yVec = np.zeros(i+1,dtype=np.float32)
		xVec = np.zeros(i+1,dtype=np.float32)

		for j in range (len(xVec)):
			xVec[j] = rVec[j]
			yVec[j] = rVec[j]*rVec[j]*(p1Vec[j] + 3*pkVec[j])

		pnVec[i] = scipy.integrate.simps(yVec,xVec) * 1./rVec[i]/rVec[i]/rVec[i]

	for i in range (len(rVec)):
		ptVec[i] = p1Vec[i] + 3*pkVec[i] - pnVec[i]
		ptVec[i] *= 0.5


cdef cal_tension(np.ndarray[float,ndim=1] rVec, np.ndarray[float,ndim=1] ptVec, np.ndarray[float,ndim=1] pnVec):
	cdef int i, j
	cdef np.ndarray[float,ndim=1] bufVec
	cdef np.ndarray[float,ndim=1] yVec
	yVec = np.zeros(len(rVec), dtype=np.float32)

	for i in range (1, len(rVec)):
		yVec[i] = pnVec[i] - ptVec[i]
	return np.trapz(yVec,rVec)

