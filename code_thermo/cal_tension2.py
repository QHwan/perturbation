import numpy as np
import sys
import scipy.integrate
import scipy.signal

iFileName = sys.argv[1]
oFileName = sys.argv[2]

iMat = np.transpose(np.loadtxt(iFileName))
rVec = iMat[0]
pkVec = iMat[1]
dFVec = iMat[2]
dEVec = iMat[3]
dSVec = iMat[4]


rMax = rVec[len(rVec)-1]

ptVec = np.zeros(len(rVec))
pnVec = np.zeros(len(rVec))
ptEVec = np.zeros(len(rVec))
pnEVec = np.zeros(len(rVec))
ptSVec = np.zeros(len(rVec))
pnSVec = np.zeros(len(rVec))

oMat = []
yVec = np.zeros(len(rVec))
xVec = np.zeros(len(rVec))
bufVec = np.zeros(len(rVec))
buf2Vec = np.zeros(len(rVec))
ptVec_new = np.zeros(len(rVec))


# Smoothing curve
#ptVec_new = scipy.signal.savgol_filter(ptVec,window_length=11,polyorder=3)


# Correcting value
'''
for i in range (len(rVec)):
	yVec[i] = ptVec_new[i]*rVec[i]
correction = np.trapz(yVec,rVec)*2/rMax/rMax
print correction
'''


pnVec[0] = ptVec[0]

for i in range (1,len(rVec)):
	yVec = np.zeros(i+1)
	xVec = np.zeros(i+1)

	for j in range (len(xVec)):
		xVec[j] = rVec[j]
		yVec[j] = rVec[j]*rVec[j]*(dFVec[j] + 3*pkVec[j])

	pnVec[i] = scipy.integrate.simps(yVec,xVec) * 1./rVec[i]/rVec[i]/rVec[i]

	for j in range (len(xVec)):
		xVec[j] = rVec[j]
		yVec[j] = rVec[j]*rVec[j]*(dEVec[j] + 3*pkVec[j])

	pnEVec[i] = scipy.integrate.simps(yVec,xVec) * 1./rVec[i]/rVec[i]/rVec[i]

	for j in range (len(xVec)):
		xVec[j] = rVec[j]
		yVec[j] = rVec[j]*rVec[j]*(dSVec[j] + 3*pkVec[j])

	pnSVec[i] = scipy.integrate.simps(yVec,xVec) * 1./rVec[i]/rVec[i]/rVec[i]

for i in range (len(rVec)):
	ptVec[i] = dFVec[i] + 3*pkVec[i] - pnVec[i]
	ptVec[i] *= 0.5

for i in range (len(rVec)):
	ptEVec[i] = dEVec[i] + 3*pkVec[i] - pnEVec[i]
	ptEVec[i] *= 0.5

for i in range (len(rVec)):
	ptSVec[i] = dSVec[i] + 3*pkVec[i] - pnSVec[i]
	ptSVec[i] *= 0.5

#pnVec[0] -= correction

# Calculation tension
yVec = np.zeros(len(rVec))
for i in range (1, len(rVec)):
	yVec[i] = pnVec[i] - ptVec[i]
print "tension = {}".format(np.trapz(yVec,rVec))

for i in range (1, len(rVec)):
	yVec[i] = pnEVec[i] - ptEVec[i]
print "tensionE = {}".format(np.trapz(yVec,rVec))

for i in range (1, len(rVec)):
	yVec[i] = pnSVec[i] - ptSVec[i]
print "tensionS = {}".format(np.trapz(yVec,rVec))

'''
pVec = np.zeros(len(ptVec))
for i in range (len(pVec)):
	pVec[i] = (pkVec[i] + (ptVec[i]-pkVec[i]) + (pnVec[i]-pkVec[i]))
'''

oMat.append(rVec)
oMat.append(ptVec)
oMat.append(pnVec)
oMat.append(ptEVec)
oMat.append(pnEVec)
oMat.append(ptSVec)
oMat.append(pnSVec)
oMat = np.transpose(oMat)

np.savetxt(oFileName,oMat,fmt='%7f')

