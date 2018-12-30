import numpy as np
import sys

iFileName = sys.argv[1]
oFileName = sys.argv[2]

iMat = np.loadtxt(iFileName)


iVec = iMat[10]

oMat = []
for i in range (len(iVec)-1):
	oMat.append([i+1, iVec[i+1]])

np.savetxt(oFileName,oMat,fmt='%e')




