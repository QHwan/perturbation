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
	cdef np.ndarray[float, ndim=1] rVec
	cdef np.ndarray[float, ndim=3] boxMat
	cdef np.ndarray[float, ndim=1] vVec
	cdef np.ndarray[float, ndim=1] dvVec

	cdef np.ndarray[float, ndim=2] LJpMat
	cdef np.ndarray[float, ndim=2] LJmMat
	cdef np.ndarray[float, ndim=2] CpMat
	cdef np.ndarray[float, ndim=2] CmMat

	cdef np.ndarray[float, ndim=1] UpVec
	cdef np.ndarray[float, ndim=1] UmVec

	cdef np.ndarray[float, ndim=1] dUVec
	cdef np.ndarray[float, ndim=1] dU2Vec
	cdef np.ndarray[float, ndim=1] dU3Vec
	cdef np.ndarray[float, ndim=1] UVec
	cdef np.ndarray[float, ndim=1] UdUVec
	cdef np.ndarray[float, ndim=1] UdU2Vec
	cdef np.ndarray[float, ndim=1] dSVec
	cdef np.ndarray[float, ndim=1] dEVec

	cdef np.ndarray[float, ndim=1] buf_dUVec
	cdef float buf_dU1

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

	cdef float *ijVec

	

	# Program starts here
	# Setting the File variable
	trrFileName = sys.argv[1]
	mFileName = sys.argv[2]
	denFileName = sys.argv[3]
	oFileName = sys.argv[4]

	mMat = np.transpose(np.loadtxt(mFileName))
	mVec = mMat[1]

	dMat = np.transpose(np.loadtxt(denFileName,dtype=np.float32))
	rVec = dMat[0]
	dVec = dMat[1]

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
	dr = rVec[1]-rVec[0]
	rMin = dr*0.5
	rMax = rVec[len(rVec)-1]


	
	

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
	CpMat = np.zeros((nH2O, nH2O), dtype=np.float32)
	CmMat = np.zeros((nH2O, nH2O), dtype=np.float32)

	UpVec = np.zeros(nH2O, dtype=np.float32)
	UmVec = np.zeros(nH2O, dtype=np.float32)

	dUVec = np.zeros(len(rVec), dtype=np.float32)
	dU2Vec = np.zeros(len(rVec), dtype=np.float32)
	dU3Vec = np.zeros(len(rVec), dtype=np.float32) 
	UVec = np.zeros(len(rVec), dtype=np.float32)
	UdUVec = np.zeros(len(rVec), dtype=np.float32)
	UdU2Vec = np.zeros(len(rVec), dtype=np.float32) 

	dSVec = np.zeros(len(rVec), dtype=np.float32)
	dEVec = np.zeros(len(rVec), dtype=np.float32)

	p1Vec = np.zeros(len(rVec), dtype=np.float32)
	p2Vec = np.zeros(len(rVec), dtype=np.float32)
	p3Vec = np.zeros(len(rVec), dtype=np.float32) 

	pkVec = np.zeros(len(rVec), dtype=np.float32)

	countVec = np.zeros(len(rVec), dtype=np.float32)

	# Setting perturbation environment
	eps = 1e-4
	epsp = eps
	#epsm = (2 - (1+epsp)**3)**(1./3.) - 1
	epsm = 0.
	print epsp, epsm
	dVp = (1+epsp)**3
	dVm = (1+epsm)**3
	dV = dVp - dVm

	Vtot = nH2O * 1/33.34
	Vp = Vtot*(1+epsp)
	Vm = Vtot*(1+epsm)

	vVec = np.zeros(len(rVec), dtype=np.float32)
	for i in range (len(rVec)):
		vVec[i] = 4*math.pi/3.*(pow(rVec[i] + dr*0.5,3) - pow(rVec[i] - dr*0.5,3))

	dvVec = np.zeros(len(rVec), dtype=np.float32)
	for i in range (len(rVec)):
		dvVec[i] = dV*vVec[i]


	
	
	# Starting read coordinates
	frame = 0
	nFrame_used = 0
	for coordMat in coordtMat:

		if frame % 1000 == 0:
			print frame

		if mVec[frame] != nH2O and nFrame != 1:
		#if mVec[frame] < 0 and nFrame != 1:
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
						#for l in range (3):
							iMat[k][0] = coordMat[apm*i+k,0]
							iMat[k][1] = coordMat[apm*i+k,1]
							iMat[k][2] = coordMat[apm*i+k,2]
							jMat[k][0] = coordMat[apm*j+k,0]
							jMat[k][1] = coordMat[apm*j+k,1]
							jMat[k][2] = coordMat[apm*j+k,2]
							
					coord_pert(iMat, bufMat, epsp)
					coord_pert(jMat, bufMat, epsp)
					
					
					LJpMat[i,j] = cal_LJ(iMat, jMat)
					CpMat[i,j] = cal_C(iMat,jMat, ijVec)


					for k in range (apm):
						#for l in range (3):
							iMat[k][0] = coordMat[apm*i+k,0]
							iMat[k][1] = coordMat[apm*i+k,1]
							iMat[k][2] = coordMat[apm*i+k,2]
							jMat[k][0] = coordMat[apm*j+k,0]
							jMat[k][1] = coordMat[apm*j+k,1]
							jMat[k][2] = coordMat[apm*j+k,2]

					#coord_pert(iMat, bufMat, epsm)
					#coord_pert(jMat, bufMat, epsm)
					
					LJmMat[i,j] = cal_LJ(iMat, jMat)
					CmMat[i,j] = cal_C(iMat,jMat,ijVec)


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

			for i in range (nH2O):
				r = sqrt(coordMat[apm*i,0]**2 + coordMat[apm*i,1]**2 + coordMat[apm*i,2]**2)


				if r>=rMin-0.5*dr and r<rMax:
					dUVec[int(r/dr)] += UpVec[i] - UmVec[i]
					dU2Vec[int(r/dr)] += (UpVec[i] - UmVec[i])**2
					dU3Vec[int(r/dr)] += (UpVec[i] - UmVec[i])**3

					UVec[int(r/dr)] += UmVec[i]
					UdUVec[int(r/dr)] += UmVec[i]*(UpVec[i]-UmVec[i])
					UdU2Vec[int(r/dr)] += UmVec[i]*(UpVec[i]-UmVec[i])*(UpVec[i]-UmVec[i])


					countVec[int(r/dr)] += 1.

				else:
					pass



	# Ensemble Average
	# Volume Normalization
	# Set pkVec
	print 'Total frame = {}'.format(nFrame)
	print 'Used frame = {}'.format(nFrame_used)

	for i in range (len(rVec)):
		pkVec[i] = dVec[i]

	'''
	for i in range (len(rVec)):
		if countVec[i] == 0:
			dUVec[i] = 0
			dU2Vec[i] = 0
			dU3Vec[i] = 0
		else:
			dUVec[i] /= countVec[i]
			dU2Vec[i] /= countVec[i]
			dU3Vec[i] /= countVec[i]
	'''
	for i in range (len(rVec)):
		dUVec[i] /= nFrame_used
		dU2Vec[i] /= nFrame_used
		dU3Vec[i] /= nFrame_used

		UVec[i] /= nFrame_used
		UdUVec[i] /= nFrame_used
		UdU2Vec[i] /= nFrame_used


	for i in range (len(rVec)):
		p1Vec[i] = -dUVec[i]
		#p2Vec[i] = 0.5/kBT*(dU2Vec[i] - dUVec[i]*dUVec[i])
		#p3Vec[i] = -1./6/kBT/kBT*(dU3Vec[i] - 3*dU2Vec[i]*dUVec[i] + 2*dUVec[i]**3)

		pkVec[i] *= kBT

		dSVec[i] = -1/kBT*(UdUVec[i]-UVec[i]*dUVec[i])
		#dSVec[i] -= -0.5/kBT*(dU2Vec[i] - dUVec[i]*dUVec[i])
		#dSVec[i] += 0.5/kBT/kBT*(UdU2Vec[i] - UVec[i]*dU2Vec[i])
		#dSVec[i] -= 1/kBT/kBT*dUVec[i]*(UdUVec[i] - UVec[i]*dUVec[i])

		dEVec[i] = dUVec[i] - 1/kBT*(UdUVec[i] - UVec[i]*dUVec[i])

		#p1Vec[i] *= pref/dvVec[i]
		#p2Vec[i] *= pref/dvVec[i]
		#p3Vec[i] *= pref/dvVec[i]

		p1Vec[i] *= pref/vVec[i]/epsp
		p2Vec[i] *= pref/vVec[i]/epsp
		p3Vec[i] *= pref/vVec[i]/epsp

		dSVec[i] *= pref/vVec[i]/epsp
		dEVec[i] *= pref/vVec[i]/epsp

		pkVec[i] *= pref
	

	

	oMat = []
	oMat.append(rVec)
	oMat.append(pkVec)
	oMat.append(p1Vec)
	oMat.append(dEVec)
	oMat.append(dSVec)
	oMat.append(pkVec + p1Vec)
	np.savetxt(oFileName,np.transpose(oMat),fmt='%e')


					
				
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


		


							






