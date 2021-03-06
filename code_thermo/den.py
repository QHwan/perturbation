import numpy as np
import sys
import math
from mdtraj.formats import XTCTrajectoryFile
from mdtraj.formats import TRRTrajectoryFile

trrFileName = sys.argv[1]
mFileName = sys.argv[2]
oFileName = sys.argv[3]

mMat = np.transpose(np.loadtxt(mFileName))
mVec = mMat[1]

with TRRTrajectoryFile(trrFileName) as trrFile:
	coordtMat, tVec, step, boxMat, lambdaVec = trrFile.read()
trrFile.close()

nFrame, nH2O, xyz = np.shape(coordtMat)
apm = 4
nH2O /= apm

L = boxMat[0][0][0]
L2 = L*0.5

# Setting PMat
dr = 0.05
rMin = dr*0.5
rMax = 4.
rVec = np.zeros(int((rMax-rMin)/dr)+1)
denVec = np.zeros(len(rVec))


for i in range (len(rVec)):
	rVec[i] = rMin + dr*i

vVec = np.zeros(len(rVec))
for i in range (len(rVec)):
	vVec[i] = 4*math.pi/3.*(pow(rMin + dr*i + dr*0.5,3) - pow(rMin + dr*i - dr*0.5,3))

frame = 0
nFrame_used = 0
for coordMat in coordtMat:
	if frame % 1000 == 0:
		print frame

	if mVec[frame] != nH2O:
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


		for i in range (nH2O):
			x = coordMat[apm*i,0]
			y = coordMat[apm*i,1]
			z = coordMat[apm*i,2]
			r = math.sqrt(x**2+y**2+z**2)

			if r>=rMin-0.5*dr and r<rMax:
				denVec[int(r/dr)] += 1
			else:
				pass


for i in range (len(rVec)):
	denVec[i] *= 1/vVec[i]/nFrame_used

oMat = []
oMat.append(rVec)
oMat.append(denVec)
oMat = np.transpose(oMat)

np.savetxt(oFileName,oMat,fmt='%7f')
