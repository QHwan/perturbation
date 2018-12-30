import numpy as np
import sys
import scipy.integrate
import scipy.signal

iFileName = sys.argv[1]
oFileName = sys.argv[2]

buf_ptMat = []
buf_pnMat = []
tensionVec = []
RsVec = []
for n in range (10):
	i2FileName = iFileName.replace(".xvg","")
	i2FileName = i2FileName + '_' + str(n) + '.xvg'
	iMat = np.transpose(np.loadtxt(i2FileName))
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


	# Calculation tension
	yVec = np.zeros(len(rVec))
	for i in range (5, len(rVec)):
		yVec[i] = pnVec[i] - ptVec[i]

	tensionVec.append(np.trapz(yVec,rVec))

	


	buf_ptMat.append(ptVec)
	buf_pnMat.append(pnVec)


buf_ptMat = np.transpose(buf_ptMat)
buf_pnMat = np.transpose(buf_pnMat)

for i in range (len(rVec)):
	pt_avg = np.average(buf_ptMat[i])
	pt_std = np.std(buf_ptMat[i])
	pn_avg = np.average(buf_pnMat[i])
	pn_std = np.std(buf_pnMat[i])
	oMat.append([rVec[i], pt_avg, pt_std, pn_avg, pn_std])

# Calculation Rs
y1Vec = np.zeros(len(rVec))
y2Vec = np.zeros(len(rVec))
dr = rVec[1]-rVec[0]
for i in range (len(rVec)-1):
	y1Vec[i] = rVec[i] * (oMat[i+1][3] - oMat[i][3])/dr
	y2Vec[i] = (oMat[i+1][3] - oMat[i][3])/dr
Rs = np.trapz(y1Vec,rVec)/np.trapz(y2Vec,rVec)
RsVec.append(Rs)

print tensionVec
print 'tension = {}, {}'.format(np.average(tensionVec), np.std(tensionVec))
print 'Rs = {}'.format(Rs)


np.savetxt(oFileName,oMat,fmt='%7f')

