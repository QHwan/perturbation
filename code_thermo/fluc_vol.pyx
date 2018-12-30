'''
Calculate potential difference from volume perturbation.

# Usage
python fluc_vol.py (Input) md.trr mc.xvg (Output) tension.xvg potential.xvg dv.xvg
'''
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
	cdef float Up, Um, U
	cdef float R

	cdef np.ndarray[float, ndim=3] coordtMat
	cdef np.ndarray[float, ndim=2] coordMat
	cdef np.ndarray[float, ndim=1] tVec
	cdef np.ndarray[float, ndim=3] boxMat
	cdef np.ndarray[float, ndim=1] vVec
	cdef np.ndarray[float, ndim=1] dvVec

	cdef np.ndarray[float, ndim=2] LJpMat
	cdef np.ndarray[float, ndim=2] LJmMat
	cdef np.ndarray[float, ndim=2] LJMat
	cdef np.ndarray[float, ndim=2] CpMat
	cdef np.ndarray[float, ndim=2] CmMat
	cdef np.ndarray[float, ndim=2] CMat

	cdef np.ndarray[float, ndim=1] UpVec
	cdef np.ndarray[float, ndim=1] UmVec
	cdef np.ndarray[float, ndim=1] UVec

	cdef np.ndarray[float, ndim=1] dU1Vec
	cdef np.ndarray[float, ndim=1] dU2Vec
	cdef np.ndarray[float, ndim=1] dU3Vec

	cdef float buf_dUp1, buf_dUp2, buf_dUp3
	cdef float buf_dUm1, buf_dUm2, buf_dUm3
	cdef float dUp1, dUp2, dUp3
	cdef float dUp1_avg, dUp2_avg, dUp3_avg
	cdef float dUm1, dUm2, dUm3
	cdef float dUm1_avg, dUm2_avg, dUm3_avg
	cdef float tension1
	cdef float tension2
	cdef float tension3

	cdef int count

	cdef np.ndarray[float, ndim=1] p1Vec
	cdef np.ndarray[float, ndim=1] p2Vec
	cdef np.ndarray[float, ndim=1] p3Vec

	cdef np.ndarray[float, ndim=1] pkVec


#	cdef np.ndarray[float, ndim=2] mMat
#	cdef np.ndarray[float, ndim=1] mVec

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
	o2FileName = sys.argv[4]

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


	# Setting potential environment
	LJpMat = np.zeros((nH2O, nH2O), dtype=np.float32)
	LJmMat = np.zeros((nH2O, nH2O), dtype=np.float32)
	LJMat = np.zeros((nH2O, nH2O), dtype=np.float32)
	CpMat = np.zeros((nH2O, nH2O), dtype=np.float32)
	CmMat = np.zeros((nH2O, nH2O), dtype=np.float32)
	CMat = np.zeros((nH2O, nH2O), dtype=np.float32)

	UpVec = np.zeros(nH2O, dtype=np.float32)
	UmVec = np.zeros(nH2O, dtype=np.float32)
	UVec = np.zeros(nH2O, dtype=np.float32)


	# Setting perturbation environment
	eps = 1e-3
	epsp = eps
	epsm = -eps

	Vtot = nH2O * 1/33.34
	R = pow(3*Vtot/4/3.141592, 1./3.)
	Vp = Vtot*(1+epsp)**3 - Vtot
	Vm = Vp

	print Vp

	dUp1 = 0.
	dUp2 = 0.
	dUp3 = 0.
	dUm1 = 0.
	dUm2 = 0.
	dUm3 = 0.



	oMat = []
	o2Mat = []
	o3Mat = []
	
	U1Vec = []
	U2Vec = []
	
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
					CpMat[i,j] = cal_C(iMat,jMat,ijVec)



					for k in range (apm):
						for l in range (3):
							iMat[k][l] = coordMat[apm*i+k,l]
							jMat[k][l] = coordMat[apm*j+k,l]
					LJMat[i,j] = cal_LJ(iMat,jMat)
					CMat[i,j] = cal_C(iMat,jMat,ijVec)



			for i in range (nH2O):
				for j in range (nH2O):
					if i>j:
						LJpMat[i,j] = LJpMat[j,i]
						CpMat[i,j] = CpMat[j,i]

						LJMat[i,j] = LJMat[j,i]
						CMat[i,j] = CMat[j,i]


			Up = 0.
			U = 0.
			for i in range (nH2O):
				for j in range (nH2O):
					Up += LJpMat[i,j]
					Up += CpMat[i,j]
					U += LJMat[i,j]
					U += CMat[i,j]

			Up /= 2
			U /= 2


			# Distribute

			buf_dUp1 = Up - U
			buf_dUp2 = (Up - U)**2
			buf_dUp3 = (Up - U)**3

			dUp1 += buf_dUp1
			dUp2 += buf_dUp2
			dUp3 += buf_dUp3

			dUp1_avg = dUp1/nFrame_used
			dUp2_avg = dUp2/nFrame_used
			dUp3_avg = dUp3/nFrame_used

			tension1 = dUp1_avg/Vp
			tension2 = -0.5/kBT*((dUp2_avg-dUp1_avg*dUp1_avg)/Vp) 
			tension3 = 1./6/kBT/kBT*((dUp3_avg - 3*dUp2_avg*dUp1_avg + 2*dUp1_avg*dUp1_avg*dUp1_avg)/Vp)

			tension1 *= pref
			tension2 *= pref
			tension3 *= pref

			#oMat.append([tVec[frame-1],buf_dU1/(Vp), -0.5/kBT*(buf_dU2-buf_dU1*buf_dU1)/Vp, 1./6/kBT/kBT*(buf_dU3 - 3*buf_dU2*buf_dU1 + 2*buf_dU1*buf_dU1*buf_dU1)])
			oMat.append([tVec[frame-1], tension1, tension2, tension3])
			U1Vec.append(buf_dUp1)
			U2Vec.append(buf_dUp2)

			o2Mat.append([tVec[frame-1], buf_dUp1, buf_dUp2, buf_dUp3])

			o3Mat.append([tVec[frame-1], Vp])

	# Ensemble Average
	# Volume Normalization
	# Set pkVec
	print 'Total frame = {}'.format(nFrame)
	print 'Used frame = {}'.format(nFrame_used)


	np.savetxt(oFileName,oMat,fmt='%e')
	np.savetxt(o2FileName,o2Mat,fmt='%e')

					
				
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


		


							






