import cython
cimport cython
import numpy as np
cimport numpy as np
import sys
import math
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
import time
from mdtraj.formats import XTCTrajectoryFile
from mdtraj.formats import TRRTrajectoryFile

from func cimport cal_LJ, cal_C, coord_pert


@cython.cdivision(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def main():

	# This line we declare the variables
	cdef int i, j, k, l
	cdef int frame, nFrame, nFrame_used, nH2O, apm
	cdef int lenCoordVec
	cdef float kBT, T
	cdef float doh, dom
	cdef float t
	cdef float comx, comy, comz
	cdef float L, L2
	cdef float rMin, rMax, dr
	cdef float x, y, z, x0, y0, z0, r
	cdef float ri, rj, rij, ri2, rj2, rij2, ririj
	cdef float Din, Dout, sDin, sDout, Rin, Rout, Rin2, Rout2
	cdef float linn, linp, loutn, loutp, ll
	cdef float la, lb, lap, lbp
	cdef float pN, pT, pK
	cdef float v
	cdef float pref
	cdef float eps, epsp, epsm, dVp, dVm, dV
	cdef float Vtot, Vp, Vm
	cdef float Up, Um

	cdef np.ndarray[float, ndim=3] coordtMat
	cdef np.ndarray[float, ndim=2] coordMat
	cdef np.ndarray[float, ndim=1] tVec
	cdef np.ndarray[float, ndim=3] boxMat
	cdef np.ndarray[float, ndim=1] vVec
	cdef np.ndarray[float, ndim=1] dvVec

	cdef np.ndarray[float, ndim=2] LJpMat
	cdef np.ndarray[float, ndim=2] LJmMat
	cdef np.ndarray[float, ndim=2] CpMat
	cdef np.ndarray[float, ndim=2] CmMat

	cdef np.ndarray[float, ndim=1] UpVec
	cdef np.ndarray[float, ndim=1] UmVec

	cdef np.ndarray[float, ndim=1] dU1Vec
	cdef np.ndarray[float, ndim=1] dU2Vec
	cdef np.ndarray[float, ndim=1] dU3Vec

	cdef np.ndarray[float, ndim=1] buf_dU1Vec
	cdef float buf_dU1
	cdef float buf_dU2

	cdef np.ndarray[float, ndim=1] p1Vec
	cdef np.ndarray[float, ndim=1] p2Vec
	cdef np.ndarray[float, ndim=1] p3Vec

	cdef np.ndarray[float, ndim=1] pkVec

	cdef np.ndarray[float, ndim=1] countVec

#	cdef np.ndarray[float, ndim=2] mMat
#	cdef np.ndarray[float, ndim=1] mVec

	cdef float **iMat 
	cdef float **jMat
	cdef float **bufMat

	cdef float **rMat

	

	# Program starts here
	# Setting the File variable
	trrFileName = sys.argv[1]
	oFileName = sys.argv[2]



	with TRRTrajectoryFile(trrFileName) as trrFile:
		coordtMat, tVec, step, boxMat, lambdaVec = trrFile.read()
	trrFile.close()

	nFrame, nH2O, xyz = np.shape(coordtMat)
	apm = 4
	nH2O /= apm

	L = boxMat[0][0][0]
	L2 = L*0.5

	pref = 1.6603	# kJ/mol/nm3 -> MPa
	T = 300.
	kBT = 2.479*T/298

	# Setting PMat
	rMin = 0.
	rMax = 3.


	
	

	# Setting iMat and jMat
	iMat = <float **> malloc(apm * sizeof(float*))
	jMat = <float **> malloc(apm * sizeof(float*))
	bufMat = <float **> malloc(apm * sizeof(float*))
	for i in range (apm):
		iMat[i] = <float *> malloc(3 * sizeof(float))
		jMat[i] = <float *> malloc(3 * sizeof(float))
		bufMat[i] = <float *> malloc(3 * sizeof(float))
	for i in range (apm):
		for j in range (3):
			iMat[i][j] = 0
			jMat[i][j] = 0
			bufMat[i][j] = 0

	rMat = <float **> malloc(apm*nH2O * sizeof(float*))
	for i in range (apm*nH2O):
		rMat[i] = <float *> malloc(3 * sizeof(float))
	for i in range (apm*nH2O):
		for j in range (3):
			rMat[i][j] = 0


	# Setting potential environment
	LJpMat = np.zeros((nH2O, nH2O), dtype=np.float32)
	LJmMat = np.zeros((nH2O, nH2O), dtype=np.float32)
	CpMat = np.zeros((nH2O, nH2O), dtype=np.float32)
	CmMat = np.zeros((nH2O, nH2O), dtype=np.float32)

	UpVec = np.zeros(nH2O, dtype=np.float32)
	UmVec = np.zeros(nH2O, dtype=np.float32)


	# Setting perturbation environment
	eps = 1e-4
	epsp = eps
	epsm = (2 - (1+epsp)**3)**(1./3.) - 1
	#epsm = 0.
	dVp = (1+epsp)**3
	dVm = (1+epsm)**3
	dV = dVp - dVm

	Vtot = nH2O * 1/33.34
	Vp = Vtot*((1+epsp)**3)
	Vm = Vtot*((1+epsm)**3)

	print Vp-Vm



	oMat = []
	
	
	# Starting read coordinates
	frame = 0
	nFrame_used = 0
	for coordMat in coordtMat:

		frame += 1
		if frame%1000==0:
			print frame


		# Consider periodic boundary
		x0 = coordMat[0,0]; y0 = coordMat[0,1]; z0 = coordMat[0,2]
		for i in range (1, nH2O):
			x = coordMat[apm*i,0]
			y = coordMat[apm*i,1]
			z = coordMat[apm*i,2]
			if x-x0 > L2:
				for j in range (apm):
					coordMat[apm*i+j,0] -= L
			elif x-x0 < -L2:
				for j in range (apm):
					coordMat[apm*i+j,0] += L
			if y-y0 > L2:
				for j in range (apm):
					coordMat[apm*i+j,1] -= L
			elif y-y0 < -L2:
				for j in range (apm):
					coordMat[apm*i+j,1] += L
			if z-z0 > L2:
				for j in range (apm):
					coordMat[apm*i+j,2] -= L
			elif z-z0 < -L2:
				for j in range (apm):
					coordMat[apm*i+j,2] += L

		# calculate com matrix using OW position
		comx = 0; comy = 0; comz = 0;
		for i in range (nH2O):
			comx += coordMat[apm*i,0]
			comy += coordMat[apm*i,1]
			comz += coordMat[apm*i,2]
		comx /= nH2O
		comy /= nH2O
		comz /= nH2O
		for i in range (apm*nH2O):
			coordMat[i,0] -= comx
			coordMat[i,1] -= comy
			coordMat[i,2] -= comz



		# Calculate Potential
		
		for i in range (nH2O):
		
			for j in range (i+1, nH2O):
				for k in range (apm):
					for l in range (3):
						iMat[k][l] = coordMat[apm*i+k,l]
						jMat[k][l] = coordMat[apm*j+k,l]
						
				coord_pert(iMat, bufMat, epsp)
				coord_pert(jMat, bufMat, epsp)
				
				
				LJpMat[i,j] = cal_LJ(iMat, jMat)
				CpMat[i,j] = cal_C(iMat,jMat)


				for k in range (apm):
					for l in range (3):
						iMat[k][l] = coordMat[apm*i+k,l]
						jMat[k][l] = coordMat[apm*j+k,l]

				coord_pert(iMat, bufMat, epsm)
				coord_pert(jMat, bufMat, epsm)
				
				LJmMat[i,j] = cal_LJ(iMat, jMat)
				CmMat[i,j] = cal_C(iMat,jMat)


		for i in range (nH2O):
			for j in range (nH2O):
				if i>j:
					LJpMat[i,j] = LJpMat[j,i]
					CpMat[i,j] = CpMat[j,i]
					LJmMat[i,j] = LJmMat[j,i]
					CmMat[i,j] = CmMat[j,i]


		for i in range (nH2O):
			Up = 0.
			Um = 0.
			for j in range (nH2O):
				Up += LJpMat[i,j]
				Up += CpMat[i,j]
				Um += LJmMat[i,j]
				Um += CmMat[i,j]

			UpVec[i] = Up/2
			UmVec[i] = Um/2

		# Distribute

		buf_dU1 = 0.
		buf_dU2 = 0.
		for i in range (nH2O):
			r = math.sqrt(coordMat[apm*i,0]**2 + coordMat[apm*i,1]**2 + coordMat[apm*i,2]**2)


			if r>=rMin and r<rMax:


				buf_dU1 += UpVec[i] - UmVec[i]
				buf_dU2 += (UpVec[i] - UmVec[i])**2


			else:
				pass


		oMat.append([tVec[frame-1],buf_dU1/(Vp-Vm), buf_dU2/(Vp-Vm)])

	# Ensemble Average
	# Volume Normalization
	# Set pkVec
	print 'Total frame = {}'.format(nFrame)
	print 'Used frame = {}'.format(nFrame_used)


	

	np.savetxt(oFileName,oMat,fmt='%e')

					
				
	# Free everything
	for i in range (apm):
		free(iMat[i])
		free(jMat[i])
		free(bufMat[i])
	free(iMat)
	free(jMat)
	free(bufMat)

	for i in range (apm*nH2O):
		free(rMat[i])
	free(rMat)


		


							






