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

import scipy.linalg

from func cimport cal_LJ, cal_C, coord_pert_area


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
	cdef float la, lb, lap, lbp
	cdef float pN, pT, pK
	cdef float v
	cdef float pref
	cdef float Vtot, Ap, Am
	cdef float R
	cdef float a, b, c

	cdef np.ndarray[float, ndim=3] coordtMat
	cdef np.ndarray[float, ndim=2] coordMat
	cdef np.ndarray[float, ndim=1] tVec
	cdef np.ndarray[float, ndim=3] boxMat
	cdef np.ndarray[float, ndim=1] vVec
	cdef np.ndarray[float, ndim=1] dvVec

	cdef np.ndarray[float, ndim=2] IMat
	cdef np.ndarray[float, ndim=2] eMat
	cdef np.ndarray[float, ndim=1] eVec

	
	cdef int count

	


	cdef float **iMat 
	cdef float **jMat
	cdef float **bufMat

	cdef float **rMat

	cdef float *ijVec


	

	# Program starts here
	# Setting the File variable
	trrFileName = sys.argv[1]
	mFileName = sys.argv[2]
	oFileName = sys.argv[3]

	mMat = np.transpose(np.loadtxt(mFileName))
	mVec = mMat[1]

	tVec = np.zeros(10, dtype=np.float32)

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

	ijVec = <float *> malloc(3 * sizeof(float))
	for i in range (3):
		ijVec[i] = 0

	IMat = np.zeros((3,3), dtype=np.float32)
	eMat = np.zeros((3,3), dtype=np.float32)
	eVec = np.zeros(3, dtype=np.float32)



	# Setting perturbation environment
	oMat = []
	
	# Starting read coordinates
	frame = 0
	nFrame_used = 0
	for coordMat in coordtMat:

		if frame%1000==0:
			print frame

		if mVec[frame] != nH2O and nFrame != 1:
			frame += 1
			pass

		else:
			frame += 1
			nFrame_used += 1

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

			# calculate of inertial tensor
			for i in range (3):
				for j in range (3):
					IMat[i,j] = 0

			for i in range (3):
				for j in range (3):
					if i==j:
						for k in range (nH2O):
							IMat[i,j] += coordMat[apm*k,0]**2 + coordMat[apm*k,1]**2 + coordMat[apm*k,2]**2 - coordMat[apm*k,i]*coordMat[apm*k,j]
					else:
						for k in range (nH2O):
							IMat[i,j] += -coordMat[apm*k,i]*coordMat[apm*k,j]

			eVec, eMat = scipy.linalg.eigh(IMat)
			print eVec, eMat; exit(1)

			#a = sqrt((5*eVec[1] + 5*eVec[2] - 5*eVec[0])/2./nH2O)
			#b = sqrt((5*eVec[0] + 5*eVec[2] - 5*eVec[1])/2./nH2O)
			#c = sqrt((5*eVec[0] + 5*eVec[1] - 5*eVec[2])/2./nH2O)
			a = 1/sqrt(eVec[0])
			b = 1/sqrt(eVec[1])
			c = 1/sqrt(eVec[2])


			#oMat.append([tVec[frame-1], a/c])
			oMat.append(a/c)

	histVec, binVec = np.histogram(oMat,bins=50)
	oMat = []
	for i in range (len(histVec)):
		oMat.append([(binVec[i]+binVec[i+1])*0.5, histVec[i]])
		


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

	free(ijVec)


		


							






