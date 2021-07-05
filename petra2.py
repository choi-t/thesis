import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import os

filepath='C:/data/P01/raw2/eh2_11004701_0'

kaxes69 = filepath+'1475' # 13 63 103 147; center of 4 Xtals
kaelastic69 = filepath+'1476'
kafeherfd69 = filepath+'1477' # TFY is slightly better
kafeexafs69 = filepath+'1478'
kacoherfd69 = filepath+'1480' # HERFD is better
kacoexafs69 = filepath+'1481'

kaxes313 = filepath+'1491' # 21 72 110 155; center of 4 Xtals
kaelastic313 = filepath+'1492'
kafeherfd313 = filepath+'1493' # TFY is slightly better
kafeexafs313 = filepath+'1494'
kacoherfd313 = filepath+'1496' # HERFD is better
kacoexafs313 = filepath+'1497'

fekbxes69 = filepath+'1505' # 34 77 118 160; center of 4 Xtals
fekbelastic69 = filepath+'1506' # [[10,476], [10,476], [10,476], [10,476]]; edges of each Xtal
fekbherfd69 = filepath+'1507' # HERFD 7090-7135 eV scanned but most of images are still below the K-edge... WHY?
fekbtfy69 = filepath+'1508'

fekbxes313 = filepath+'1512' # 32 74 116 158; center of 4 Xtals
fekbelastic313 = filepath+'1513' # [[10,476], [10,476], [10,476], [10,476]]; edges of each Xtal
fekbherfd313 = filepath+'1514' # HERFD 7090-7135 eV scanned but most of images are still below the K-edge... WHY?
fekbtfy313 = filepath+'1515'

cokbxes69 = filepath+'1445' # 32 76 123 164; center of 4 Xtals
cokbelastic69 = filepath+'1446' # [[78,428], [29,379], [127,477], [120,470]]; edges of each Xtal
cokbherfd69 = filepath+'1447' # TFY is better; because it's from Kbeta
cokbfexas69 = filepath+'1454' # Fe TFY XAS
cokbexafs69 = filepath+'1448'

cokbherfd69_2 = filepath+'1450' # TFY is better; because it's from Kbeta
cokbfexas69_2 = filepath+'1455' # Fe TFY XAS
cokbexafs69_2 = filepath+'1451'

cokbxes313 = filepath+'1437' # 29 72 119 160; center of 4 Xtals
cokbelastic313 = filepath+'1438' # [[78,428], [31,381], [127,477], [120,470]]; edges of each Xtal
cokbherfd313 = filepath+'1439' # TFY is better; because it's from Kbeta
cokbfexas313 = filepath+'1460' # Fe TFY XAS

cokbxes313_2 = filepath+'1428' # 42 87 134 174; center of 4 Xtals
cokbelastic313_2 = filepath+'1429' # [[79,429], [32,382], [127,477], [120,470]]; edges of each Xtal
cokbherfd313_2 = filepath+'1430' # TFY is better; because it's from Kbeta
cokbfexas313_2 = filepath+'1461' # Fe TFY XAS

copyxes = filepath+'1383' # 35 79 126 166; center of 4 Xtals
copyelastic = filepath+'1384' # [[78,428], [29,379], [127,477], [120,470]]; edges of each Xtal
copyherfd = filepath+'1385' # TFY is better; because it's from Kbeta
copyexafs = filepath+'1386'

copyxes2 = filepath+'1378' # 35 79 126 166; center of 4 Xtals
copyelastic2 = filepath+'1379' # [[78,428], [29,379], [127,477], [120,470]]; edges of each Xtal
copyherfd2 = filepath+'1380' # TFY is better; because it's from Kbeta
copyexafs2 = filepath+'1381'

fefoil = filepath+'1525.fio' # mono 0.50467 eV Higher than APS; really??
fefoil_ref = 'C:/data/P01/raw2/fefoil.txt'
cofoil = filepath+'1526.fio' # mono 0.75374 eV Higher than APS; really??
cofoil_ref = 'C:/data/P01/raw2/cofoil.txt'
fefoilkbherfd = filepath+'1525'

cokbxes313_50mM = filepath+'1390' # 30 74 121 161; center of 4 Xtals
cokbelastic313_50mM = filepath+'1391' # [[80,430], [32,382], [127,477], [120,470]]; edges of each Xtal
cokbherfd313_50mM = filepath+'1392' # TFY is better; because it's from Kbeta

cokbxes313_50mM_2 = filepath+'1395' # 30 74 121 161; center of 4 Xtals
cokbelastic313_50mM_2 = filepath+'1396' # [[80,430], [32,382], [127,477], [120,470]]; edges of each Xtal
cokbherfd313_50mM_2 = filepath+'1397' # TFY is better; because it's from Kbeta

cokbxes69_50mM = filepath+'1415' # 32 76 123 164; center of 4 Xtals
cokbelastic69_50mM = filepath+'1416' # [[78,428], [29,379], [127,477], [120,470]]; edges of each Xtal
cokbherfd69_50mM = filepath+'1417' # TFY is better; because it's from Kbeta

elines = [[35, 79, 126, 166], [29, 72, 119, 160], [32, 76, 123, 164], [21, 72, 110, 155], [13, 63, 103, 147], [32, 74, 116, 158], [34, 77, 118, 160], [35, 79, 126, 166], [42, 87, 134, 174], [32, 76, 123, 164], [30, 74, 121, 161], [30, 74, 121, 161], [32, 76, 123, 164], [43, 85, 126, 167]]
crystal = [[[78,428], [29,379], [127,477], [120,470]], [[78,428], [31,381], [127,477], [120,470]], [[78,428], [29,379], [127,477], [120,470]], None, None, [[10,476], [10,476], [10,476], [10,476]], [[10,476], [10,476], [10,476], [10,476]], [[78,428], [29,379], [127,477], [120,470]], [[79,429], [32,382], [127,477], [120,470]], [[78,428], [29,379], [127,477], [120,470]], [[80,430], [32,382], [127,477], [120,470]], [[80,430], [32,382], [127,477], [120,470]], [[78,428], [29,379], [127,477], [120,470]], [[10,476], [10,476], [10,476], [10,476]]]
dataset = [[copyxes, copyelastic, copyherfd, copyexafs], [cokbxes313, cokbelastic313, cokbherfd313, cokbfexas313], [cokbxes69, cokbelastic69, cokbherfd69, cokbfexas69, cokbexafs69], [kaxes313, kaelastic313, kafeherfd313, kacoherfd313, kafeexafs313, kacoexafs313], [kaxes69, kaelastic69, kafeherfd69, kacoherfd69, kafeexafs69, kacoexafs69], [fekbxes313, fekbelastic313, fekbherfd313, fekbtfy313], [fekbxes69, fekbelastic69, fekbherfd69, fekbtfy69], [copyxes2, copyelastic2, copyherfd2, copyexafs2], [cokbxes313_2, cokbelastic313_2, cokbherfd313_2, cokbfexas313_2], [None, None, cokbherfd69_2, cokbfexas69_2, cokbexafs69_2], [cokbxes313_50mM, cokbelastic313_50mM, cokbherfd313_50mM], [cokbxes313_50mM_2, cokbelastic313_50mM_2, cokbherfd313_50mM_2], [cokbxes69_50mM, cokbelastic69_50mM, cokbherfd69_50mM], [None, None, fefoilkbherfd]]
filedic = {}

#newdic = {}
#newset = []
#newemit = [None]
#saved = 'C:/data/P01/test.txt'

class getdata:
	def __init__(self, sample, address):
		filedat = address + '/pilatus_100k/'
		filefio = address + '.fio'

		filedir = os.listdir(filedat)
		self.dset = [None]*len(filedir)
		temp_dset = [None]*len(filedir)

		for i, file in enumerate(filedir):
			fullpath = os.path.join(filedat, file)
			temp_dset[i] = misc.imread(fullpath)

		self.fset = np.loadtxt(filefio, skiprows = 107, usecols = (3,4,5,6,7)).transpose()
		self.fiox = self.fset[0] # energy axis (mono) of the measurement = # of files
		self.i0 = self.fset[1] # eh1 ion chamber I0, pin3 I cannot believe since it's not linear to ion chamber value
		self.pin3 = self.fset[4]
		test_i1 = 1-(self.fset[3]/self.i0) # transmission mode, pin2
		test_tfy = self.fset[2]/self.i0 # tfy pin1

		#plt.plot(self.fset[1],self.fset[4],'b')
		#plt.show()

		self.i1 = test_i1/max(test_i1) # Normalization?
		self.tfy = test_tfy/max(test_tfy)
		#plt.plot(self.fiox,self.i1,'b')
		#plt.show()
		#plt.plot(self.fiox,self.tfy,'b')
		#plt.show()

		self.bg = [None]*len(filedir) # mostly electronic noise or background scattered photon along x-axis

		for j, scan in enumerate(temp_dset):
			self.bg[j] = np.average(scan[185:-5], axis=0) # background noise
			self.dset[j] = scan - self.bg[j] # subtract
		'''
		if sample > 12:
			plt.imshow(self.dset[-1], cmap='terrain')
			plt.show()
		'''
		filedic[sample].append('dset_nbg')

	def elastic(self, sample, ka): # FWHM of Si(111) mono ~1 eV
		ey = elines[sample]
		emity = [None]*len(ey)

		for k, num in enumerate(ey):
			if ka == 0:
				temp = [max(a) for a in self.dset[14][num-5:num+5]] # max y within each emission line
			elif k < 2:
				temp = [max(a) for a in self.dset[75][num-5:num+5]] # Co Kalpha max y
			else:
				temp = [max(a) for a in self.dset[21][num-5:num+5]] # Fe Kalpha max y

			emity[k] = (num-5) + temp.index(max(temp)) # max y coordinate

		if ka == 0:
			# 2d array of signal x position for each elastic scan
			emitx = [[list(scan[numb]).index(max(list(scan[numb]))) for scan in self.dset[12:16]] for numb in emity]
			'''
			for scan in self.dset[12:16]:
				plt.imshow(scan, cmap='terrain', aspect='auto')
				plt.show()
			'''
		else:
			# Co Kalpha 2d array of signal x position
			# Fe Kalpha 2d array of signal x position
			emitx = [[list(scan[numb]).index(max(list(scan[numb]))) for scan in self.dset[72:78]] for numb in emity[:2]] + [[list(scan[numb]).index(max(list(scan[numb]))) for scan in self.dset[18:24]] for numb in emity[2:]]

		self.eset = [None]*len(emity) # linear fit of each emission line emitx to fio file, and calibrated x-axis
		for t, pixel in enumerate(emitx):
			linearfit = np.vstack([pixel, np.ones(len(pixel))]).T
			if ka == 0:
				o, d = np.linalg.lstsq(linearfit, self.fiox[12:16])[0]
			elif t < 2:
				o, d = np.linalg.lstsq(linearfit, self.fiox[72:78])[0]
			else:
				o, d = np.linalg.lstsq(linearfit, self.fiox[18:24])[0]

			self.eset[t] = [n*o + d for n in range(487)]

		filedic[sample].append('elastic_calibration')

		if ka == 0:
			self.xesx = self.eset[2] # standard x-axis; 3rd Xtal as a reference
			'''
			for scan in self.dset[12:16]:
				plt.plot(self.eset[0], np.average(scan[emity[0]-8:emity[0]+8], axis=0))
				plt.plot(self.eset[1], np.average(scan[emity[1]-8:emity[1]+8], axis=0))
				plt.plot(self.eset[2], np.average(scan[emity[2]-8:emity[2]+8], axis=0))
				plt.plot(self.eset[3], np.average(scan[emity[3]-8:emity[3]+8], axis=0))
				plt.show()
			'''
		else:
			self.fexesx = self.eset[3] # standard Fe Kalpha x-axis; 4th Xtal as a reference
			self.coxesx = self.eset[1] # standard Co Kalpha x-axis; 2nd Xtal as a reference

	def xes(self, sample, ka):
		ey = elines[sample]
		emity = [(num-5) + [max(a) for a in self.dset[1][num-5:num+5]].index(max([max(a) for a in self.dset[1][num-5:num+5]])) for num in ey] # max y within each emission line

		self.emitset = np.average([[np.average(scan[numb-8:numb+8], axis=0) for scan in self.dset] for numb in emity], axis=1) # averaged emission spectra from each crystal
		'''
		for spec in self.emitset:
			plt.plot(spec)

		plt.show()
		'''
		if ka == 0:
			xesy = self.emitset[2] # standard y-axis; this Xtal must be used later for x-axis
			for z, scan2 in enumerate(self.emitset):
				if z != 2:
					offset = list(self.emitset[2]).index(max(list(self.emitset[2]))) - list(scan2).index(max(list(scan2)))
					temp3 = np.concatenate((xesy[:crystal[sample][z][0]+offset], (scan2[crystal[sample][z][0]:crystal[sample][z][1]]+xesy[crystal[sample][z][0]+offset:crystal[sample][z][1]+offset])/2))
					temp4 = np.concatenate((temp3, xesy[crystal[sample][z][1]+offset:])) # all emission lines from each crystal overlapped
					xesy = temp4
					'''
					plt.plot(scan2[crystal[sample][z][0]:crystal[sample][z][1]], 'b')
					plt.plot(self.emitset[2][crystal[sample][z][0]+offset:crystal[sample][z][1]+offset], 'r')
					plt.show()
					'''
			self.xesy=xesy/np.trapz(xesy) # spectrum area normalization

		else:
			feoffset = list(self.emitset[3]).index(max(list(self.emitset[3]))) - list(self.emitset[2]).index(max(list(self.emitset[2])))
			if feoffset > 0:
				fexesy = np.concatenate((self.emitset[3][:feoffset], (self.emitset[2][:-feoffset] + self.emitset[3][feoffset:])/2)) # overlap of 2 emission lines
			else:
				fexesy = np.concatenate(((self.emitset[3][:feoffset] + self.emitset[2][-feoffset:])/2, self.emitset[3][feoffset:])) # overlap of 2 emission lines

			self.fexesy = fexesy/np.trapz(fexesy) # spectrum area normalization

			cooffset = list(self.emitset[1]).index(max(list(self.emitset[1]))) - list(self.emitset[0]).index(max(list(self.emitset[0])))
			if cooffset > 0:
				coxesy = np.concatenate((self.emitset[1][:cooffset], (self.emitset[0][:-cooffset] + self.emitset[1][cooffset:])/2)) # overlap of 2 emission lines
			else:
				coxesy = np.concatenate(((self.emitset[1][:cooffset] + self.emitset[0][-cooffset:])/2, self.emitset[1][cooffset:])) # overlap of 2 emission lines

			self.coxesy = coxesy/np.trapz(coxesy) # spectrum area normalization

		filedic[sample].append('xes_areanorm')

	def herfd2d(self, sample, ka):
		ey = elines[sample]
		emity = [(num-5) + [max(a) for a in self.dset[-5][num-5:num+5]].index(max([max(a) for a in self.dset[-5][num-5:num+5]])) for num in ey] # max y within each emission line

		self.herfdset = [None]*len(self.dset)
		self.feherfd = [None]*len(self.dset)
		self.coherfd = [None]*len(self.dset)

		offset = [list(self.dset[-5][emity[2]]).index(max(list(self.dset[-5][emity[2]]))) - list(self.dset[-5][e]).index(max(list(self.dset[-5][e]))) for e in emity]
		feoffset = list(self.dset[-5][emity[3]]).index(max(list(self.dset[-5][emity[3]]))) - list(self.dset[-5][emity[2]]).index(max(list(self.dset[-5][emity[2]])))
		cooffset = list(self.dset[-5][emity[1]]).index(max(list(self.dset[-5][emity[1]]))) - list(self.dset[-5][emity[0]]).index(max(list(self.dset[-5][emity[0]])))

		for y, scan in enumerate(self.dset):
			temp2 = [np.average(scan[numb-8:numb+8], axis=0) for numb in emity] # each emission line will be overlapped and then added into HERFD 2d image

			if ka == 0:
				herfdset = temp2[2] # standard y-axis; this Xtal must be used later for x-axis
				for c, scan2 in enumerate(temp2):
					if c != 2:
						temp3 = np.concatenate((herfdset[:crystal[sample][c][0]+offset[c]], (scan2[crystal[sample][c][0]:crystal[sample][c][1]]+herfdset[crystal[sample][c][0]+offset[c]:crystal[sample][c][1]+offset[c]])/2))
						temp4 = np.concatenate((temp3, herfdset[crystal[sample][c][1]+offset[c]:]))
						herfdset = temp4

				self.herfdset[y]=herfdset*10e3/self.i0[y] # i0 normalization

			elif ka == 1:
				if feoffset > 0:
					feherfd = np.concatenate((temp2[3][:feoffset], (temp2[2][:-feoffset] + temp2[3][feoffset:])/2)) # overlap of 2 emission lines
				else:
					feherfd = np.concatenate(((temp2[3][:feoffset] + temp2[2][-feoffset:])/2, temp2[3][feoffset:])) # overlap of 2 emission lines

				self.feherfd[y] = feherfd*10e3/self.i0[y] # i0 normalization

			else:
				if cooffset > 0:
					coherfd = np.concatenate((temp2[1][:cooffset], (temp2[0][:-cooffset] + temp2[1][cooffset:])/2)) # overlap of 2 emission lines
				else:
					coherfd = np.concatenate(((temp2[1][:cooffset] + temp2[0][-cooffset:])/2, temp2[1][cooffset:])) # overlap of 2 emission lines

				self.coherfd[y] = coherfd*10e3/self.i0[y] # i0 normalization
		'''
		if ka == 1:
			plt.imshow(self.feherfd, cmap='terrain', aspect="auto")
			plt.show()
		elif ka == 2:
			plt.imshow(self.coherfd, cmap='terrain', aspect="auto")
			plt.show()
		else:
			plt.imshow(self.herfdset[15:], cmap='terrain', aspect="auto")
			plt.show()
		'''
		filedic[sample].append('herfd2d_i0norm')

def foil(data, ref, ind):
	if ind == 0:
		dset = np.loadtxt(data, skiprows = 107, usecols = (3,4,5,6,7)).transpose() # Fe 3,5,6,7 Co 2,4,5,6
	else:
		dset = np.loadtxt(data, skiprows = 107, usecols = (2,3,4,5,6)).transpose()

	refset = np.loadtxt(ref).transpose()

	dx = dset[0][1:]-dset[0][:-1]
	tfy,i0,i1 = dset[2],dset[1],dset[3]
	dy = (tfy/i0)[1:]-(tfy/i0)[:-1]
	datamax = dset[0][list(dy/dx).index(max(list(dy/dx)))]

	rx = refset[0][1:]-refset[0][:-1]
	transm,i02 = refset[2],refset[1]
	ry = (1-(transm/i02))[1:]-(1-(transm/i02))[:-1]
	refmax = refset[0][list(ry/rx).index(max(list(ry/rx)))]

	#np.savetxt('C:/data/P01/Co_foil_TFY_XANES_P01.txt',(dset[0], (tfy/i0)/max(tfy/i0)))
	plt.plot(dset[0], (tfy/i0)/max(tfy/i0), 'r-', label='PETRA P01 Nov.2018')
	plt.plot(refset[0], (1-(transm/i02))/max(1-(transm/i02)), 'b-', label='APS reference')
	plt.legend(loc='upper right',ncol=1)
	plt.show()

	return datamax, refmax, datamax-refmax


def gui():
	print("Hi, welcome to PILATUS 100k data analysis tool.")
	print("\nCurrently we have those items:\n(Nov 2018)\n0. (Co Kbeta) Co(III)py\n1. (Co Kbeta) Fe-Co 313\n2. (Co Kbeta) Fe-Co 69\n3. (Co, Fe Kalpha) Fe-Co 313\n4. (Co, Fe Kalpha) Fe-Co 69\n5. (Fe Kbeta) Fe-Co 313\n6. (Fe Kbeta) Fe-Co 69\n7. (Co Kbeta) Co(III)py_2\n8. (Co Kbeta) Fe-Co 313_2\n9. (Co Kbeta) Fe-Co 69_2\n10. (Co Kbeta) Fe-Co 313_50mM\n11. (Co Kbeta) Fe-Co 313_50mM_2\n12. (Co Kbeta) Fe-Co 69_50mM\n13. (Fe Kbeta) Fe_foil\nStart analysis...\n")
	sample = 0
	while sample < len(dataset):
		filedic[sample] = ['xes', 'elastic', 'feherfd', 'coherfd', 'feexafs', 'coexafs']

		if sample == 3 or sample == 4:
			filedic[sample][0] = getdata(sample, dataset[sample][0])
			filedic[sample][0].xes(sample, 1)
			print("{} sample XES loaded.".format(str(sample)))
		elif sample != 9 and sample != 13:
			filedic[sample][0] = getdata(sample, dataset[sample][0])
			filedic[sample][0].xes(sample, 0)
			print("{} sample XES loaded.".format(str(sample)))

		if sample == 3 or sample == 4:
			filedic[sample][1] = getdata(sample, dataset[sample][1])
			filedic[sample][1].elastic(sample, 1)
			print("{} sample elastic scan calibrated.".format(str(sample)))
		elif sample != 9 and sample != 13:
			filedic[sample][1] = getdata(sample, dataset[sample][1])
			filedic[sample][1].elastic(sample, 0)
			print("{} sample elastic scan calibrated.".format(str(sample)))

		if sample == 3 or sample == 4:
			filedic[sample][2] = getdata(sample, dataset[sample][2])
			filedic[sample][2].herfd2d(sample, 1)
			print("{} sample Fe HERFD loaded.".format(str(sample)))
		elif sample != 5 and sample != 6:
			filedic[sample][2] = getdata(sample, dataset[sample][2])
			filedic[sample][2].herfd2d(sample, 0)
			print("{} sample HERFD loaded.".format(str(sample)))
		else:
			filedic[sample][2] = getdata(sample, dataset[sample][2])
			#for scan in filedic[sample][2].dset:
			#	plt.imshow(scan, cmap='terrain')
			#	plt.show()
			print("{} sample Fe HERFD loaded.".format(str(sample)))

		if sample == 1 or sample == 2 or sample == 5 or sample == 6 or sample == 8 or sample == 9:
			filedic[sample][3] = getdata(sample, dataset[sample][3])
			print("{} sample Fe XAS loaded.".format(str(sample)))
		elif sample == 3 or sample == 4:
			filedic[sample][3] = getdata(sample, dataset[sample][3])
			filedic[sample][3].herfd2d(sample, 2)
			print("{} sample Co HERFD loaded.".format(str(sample)))
		elif sample == 0 or sample == 7:
			filedic[sample][3] = getdata(sample, dataset[sample][3])
			filedic[sample][3].herfd2d(sample, 0)
			print("{} sample EXAFS loaded.".format(str(sample)))

		if sample == 3 or sample == 4:
			filedic[sample][4] = getdata(sample, dataset[sample][4])
			filedic[sample][4].herfd2d(sample, 1)
			print("{} sample Fe EXAFS HERFD loaded.".format(str(sample)))
			filedic[sample][5] = getdata(sample, dataset[sample][5])
			filedic[sample][5].herfd2d(sample, 2)
			print("{} sample Co EXAFS HERFD loaded.".format(str(sample)))
		elif sample == 2 or sample == 9:
			filedic[sample][4] = getdata(sample, dataset[sample][4])
			filedic[sample][4].herfd2d(sample, 0)
			print("{} sample EXAFS loaded.".format(str(sample)))

		sample += 1

	i = 0
	while i < 1:
		opt = int(input("\nPlease select an option\n0. Current datalist\n1. Plotting data\n2. New samples\n3. Saving data\n4. Exit\n"))
		if opt == 0:
			print("\nWe have currently this datalist.")
			print(filedic)
		elif opt == 1:
			plot = [int(x) for x in input("\nPlease input sample# and plot method\n0. Fe foil\n1. Co foil\n2. XES\n3. XES_all_images\n4. XES_I0\n5. XES_background\n6. XES_each_crystal\n7. Elastic_scan_images\n8. HERFD\n9. HERFD_all_images\n10. HERFD_I0\n11. HERFD_scan_TFY\n12. HERFD_scan_Transmission\n13. HERFD_background\n14. Fe_TFY_XAS\n15. XES_pin3_vs._I0,I1,Itfy\n16. HERFD_pin3_vs._I0\n").split()]
			if plot[1] == 0:
				print("\nFe Kedges (P01, APS, difference): {}".format(foil(fefoil, fefoil_ref, 0)))
			elif plot[1] == 1:
				print("\nCo Kedges (P01, APS, difference): {}".format(foil(cofoil, cofoil_ref, 1)))
			else:
				plotting(plot[0], plot[1])
		elif opt == 2:
			print("\nCurrently we have those items in newset: ")
			newdic[0] = ['raw']
			newdic[0][0] = getdata(0, newset[0][0])
			print('There come all images recorded')
			for image in newdic[0][0].dset:
				plt.imshow(image)
				plt.show()

			method = int(input("Which method do you want to use:\n0. XES\n1. elastic\n2. HERFD\n"))
			print('Choose y values of each line...')
			if method == 0:
				plt.imshow(self.dset[1])
			elif method == 1:
				plt.imshow(self.dset[14])
			elif method == 2:
				plt.imshow(self.dset[85])
			plt.show()
			newemit[0] = [int(x) for x in input("Put y values of each line: ").split()]
		elif opt == 3:
			datafile = np.array([filedic[6][1].xesx, filedic[6][0].xesy])
			datafile = datafile.T
			with open(saved, "w") as output:
				np.savetxt(output, datafile, fmt=['%.7f','%.7f'])
			print("Data saved.")
		else: i = 1


def plotting(address, method):
	if method == 2 and address < 3:
		#np.savetxt('C:/data/P01/CoKb_moreshots_cobaloxime_Fe-Co_ref.txt',(filedic[0][1].xesx,(filedic[0][0].xesy+filedic[7][0].xesy)/2,filedic[2][1].xesx,(filedic[2][0].xesy+filedic[12][0].xesy)/2))
		plt.plot(filedic[0][1].xesx, (filedic[0][0].xesy+filedic[7][0].xesy)/2, 'g-', label='(Co Kbeta) Co(III)py')
		plt.plot(filedic[1][1].xesx, (filedic[1][0].xesy+filedic[10][0].xesy+filedic[11][0].xesy)/3, 'r-', label='(Co Kbeta) Fe-Co 313 bad')
		plt.plot(filedic[2][1].xesx, (filedic[2][0].xesy+filedic[12][0].xesy)/2, 'b-', label='(Co Kbeta) Fe-Co 69 pure')
		#plt.plot(filedic[10][1].xesx, filedic[10][0].xesy, 'b-', label='(Co Kbeta) Co(III)py_2')
		plt.legend(loc='upper right',ncol=1)
		plt.show()

	elif method == 2 and address < 5:
		#np.savetxt('C:/data/P01/FeKa_Fe-Co_ref.txt',(filedic[4][1].fexesx,filedic[4][0].fexesy))
		#np.savetxt('C:/data/P01/CoKa_Fe-Co_ref.txt',(filedic[4][1].coxesx,filedic[4][0].coxesy))
		plt.plot(filedic[3][1].fexesx, filedic[3][0].fexesy, 'r-', label='(Fe Kalpha) Fe-Co 313 bad')
		plt.plot(filedic[4][1].fexesx, filedic[4][0].fexesy, 'b-', label='(Fe Kalpha) Fe-Co 69 pure')
		plt.legend(loc='upper right',ncol=1)
		plt.show()
		plt.plot(filedic[3][1].coxesx, filedic[3][0].coxesy, 'r-', label='(Co Kalpha) Fe-Co 313 bad')
		plt.plot(filedic[4][1].coxesx, filedic[4][0].coxesy, 'b-', label='(Co Kalpha) Fe-Co 69 pure')
		plt.legend(loc='upper right',ncol=1)
		plt.show()

	elif method == 2 and address < 7:
		#np.savetxt('C:/data/P01/FeKb_Fe-Co_ref.txt',(filedic[6][1].xesx,filedic[6][0].xesy))
		plt.plot(filedic[5][1].xesx, filedic[5][0].xesy, 'r-', label='(Fe Kbeta) Fe-Co 313')
		plt.plot(filedic[6][1].xesx, filedic[6][0].xesy, 'b-', label='(Fe Kbeta) Fe-Co 69')
		plt.legend(loc='upper right',ncol=1)
		plt.show()

	elif method == 3 and address != 9 and address != 13:
		for image in filedic[address][0].dset:
			plt.imshow(image)
			plt.show()

	elif method == 4 and address != 9 and address != 13:
		plt.plot(filedic[address][0].i0, 'r-')
		plt.show()

	elif method == 5 and address != 9 and address != 13:
		for scan in filedic[address][0].bg:
			plt.plot(scan)

		plt.show()

	elif method == 6 and address != 9 and address != 13:
		for n, scan in enumerate(filedic[address][0].emitset):
			plt.plot(filedic[address][1].eset[n], scan)

		plt.show()

	elif method == 7 and address != 9 and address != 13:
		for image in filedic[address][1].dset:
			plt.imshow(image)
			plt.show()

	elif method == 8 and (address < 3 or address > 6):
		print("Enter x-axis range to integrate...\n")
		#np.savetxt('C:/data/P01/CoKb_HERFD2d_Fe-Co_69_moreshots.txt',(np.array(filedic[address][2].herfdset[15:])+np.array(filedic[address+7][2].herfdset[15:])))
		#np.savetxt('C:/data/P01/CoKb_HERFD2d_Fe-Co_69_moreshots_yaxis.txt',(filedic[address][2].fiox[15:]))
		#np.savetxt('C:/data/P01/CoKb_HERFD2d_Fe-Co_69_moreshots_xaxis.txt',(filedic[address][1].xesx))
		plt.imshow(filedic[address][2].herfdset[15:], cmap='terrain', aspect="auto")
		plt.show()
		#plt.imshow(np.array(filedic[address][2].herfdset[15:])/max(filedic[address][2].herfdset[69])-np.array(filedic[address+10][2].herfdset[15:])/max(filedic[address+10][2].herfdset[69]), cmap='terrain', aspect="auto")
		#plt.show()
		if str(input("Do you want to integrate?(y/n) ")) == 'y':
			enterx=[int(v) for v in input("Where to where? ").split()]
			plt.plot(filedic[address][2].fiox, np.average(np.transpose(filedic[address][2].herfdset)[enterx[0]:enterx[1]], axis=0), label='(Co Kbeta) HERFD')
			#plt.plot(filedic[1][2].fiox, filedic[1][2].herfdy, 'b-', label='(Co Kbeta) Fe-Co 313')
			#plt.plot(filedic[2][2].fiox, filedic[2][2].herfdy, 'g-', label='(Co Kbeta) Fe-Co 69')
			plt.legend(loc='upper left',ncol=1)
			plt.show()

	elif method == 8 and address < 5:
		print("Enter x-axis range to integrate...\n")
		#np.savetxt('C:/data/P01/FeKa_HERFD2d_extended_Fe-Co69.txt',(list(filedic[address][2].feherfd)+list(filedic[address][4].feherfd)))
		#np.savetxt('C:/data/P01/FeKa_HERFD2d_extended_Fe-Co69_yaxis.txt',(list(filedic[address][2].fiox)+list(filedic[address][4].fiox)))
		#np.savetxt('C:/data/P01/FeKa_HERFD2d_extended_Fe-Co69_xaxis.txt',(filedic[address][1].fexesx))
		plt.imshow(list(filedic[address][2].feherfd)+list(filedic[address][4].feherfd), cmap='terrain', aspect="auto")
		plt.show()
		if str(input("Do you want to integrate?(y/n) ")) == 'y':
			enterx=[int(v) for v in input("Where to where? ").split()]
			plt.plot(list(filedic[address][2].fiox)+list(filedic[address][4].fiox), np.average(np.transpose(list(filedic[address][2].feherfd)+list(filedic[address][4].feherfd))[enterx[0]:enterx[1]], axis=0), label='(Fe Kalpha) HERFD')
			plt.legend(loc='upper left',ncol=1)
			plt.show()

		print("Enter x-axis range to integrate...\n")
		#np.savetxt('C:/data/P01/CoKa_HERFD2d_extended_Fe-Co69.txt',(list(filedic[address][3].coherfd)+list(filedic[address][5].coherfd)))
		#np.savetxt('C:/data/P01/CoKa_HERFD2d_extended_Fe-Co69_yaxis.txt',(list(filedic[address][3].fiox)+list(filedic[address][5].fiox)))
		#np.savetxt('C:/data/P01/CoKa_HERFD2d_extended_Fe-Co69_xaxis.txt',(filedic[address][1].coxesx))
		plt.imshow(list(filedic[address][3].coherfd)+list(filedic[address][5].coherfd), cmap='terrain', aspect="auto")
		plt.show()
		if str(input("Do you want to integrate?(y/n) ")) == 'y':
			enterx=[int(v) for v in input("Where to where? ").split()]
			#np.savetxt('C:/data/P01/CoKa1_HERFDXANES_Fe-Co69.txt',(filedic[address][3].fiox, np.average(np.transpose(filedic[address][3].coherfd)[enterx[0]:enterx[1]], axis=0)))
			plt.plot(list(filedic[address][3].fiox)+list(filedic[address][5].fiox), np.average(np.transpose(list(filedic[address][3].coherfd)+list(filedic[address][5].coherfd))[enterx[0]:enterx[1]], axis=0), label='(Co Kalpha) HERFD')
			plt.legend(loc='upper left',ncol=1)
			plt.show()

	elif method == 8 and address < 7:
		for scan in filedic[address][2].dset:
			plt.imshow(scan, cmap='terrain')
			plt.show()

	elif method == 9:
		print("HERFD PILATUS images...\n")
		for image in filedic[address][2].dset:
			plt.imshow(image)
			plt.show()

		if address == 3 or address == 4:
			print("Co HERFD PILATUS images...\n")
			for image in filedic[address][3].dset:
				plt.imshow(image)
				plt.show()

	elif method == 10:
		print("HERFD I0...")
		plt.plot(filedic[address][2].fiox, filedic[address][2].i0, 'r-')
		plt.show()

		if address == 3 or address == 4:
			print("Co HERFD I0...")
			plt.plot(filedic[address][3].fiox, filedic[address][3].i0, 'r-')
			plt.show()

	elif method == 11 and address < 3:
		#np.savetxt('C:/data/P01/Co_Kedge_XANES_TFY_moreshots_cobaloxime_Fe-Co313_Fe-Co69.txt',(filedic[0][2].fiox,(filedic[0][2].tfy+filedic[7][2].tfy)/2,filedic[1][2].fiox,(filedic[1][2].tfy+filedic[8][2].tfy)/2,filedic[2][2].fiox,(filedic[2][2].tfy+filedic[9][2].tfy)))
		#plt.plot(list(filedic[7][2].fiox)+list(filedic[7][3].fiox), list((filedic[0][2].tfy+filedic[7][2].tfy)/2)+list(filedic[7][3].tfy), 'g-', label='(Co Kbeta) Co(III)py')
		plt.plot(filedic[0][2].fiox, (filedic[0][2].tfy+filedic[7][2].tfy)/2, 'g-', label='(Co Kbeta) Co(III)py')
		plt.plot(filedic[1][2].fiox, (filedic[1][2].tfy+filedic[8][2].tfy)/2, 'r-', label='(Co Kbeta) Fe-Co 313 bad')
		plt.plot(filedic[2][2].fiox, (filedic[2][2].tfy+filedic[9][2].tfy)/2, 'b-', label='(Co Kbeta) Fe-Co 69 pure')
		#plt.plot(filedic[12][2].fiox, filedic[12][2].tfy, 'm-', label='(Co Kbeta) Fe-Co 313 50mM')
		plt.legend(loc='upper left',ncol=1)
		plt.show()

	elif method == 11 and address < 5:
		#np.savetxt('C:/data/P01/Fe_Kedge_XANES_TFY_Fe-Co313_Fe-Co69_FeCoKa.txt',(filedic[3][2].fiox,filedic[3][2].tfy,filedic[4][2].fiox,filedic[4][2].tfy))
		plt.plot(list(filedic[3][2].fiox)+list(filedic[3][4].fiox), list(filedic[3][2].tfy)+list(filedic[3][4].tfy), 'r-', label='(Fe Kalpha) Fe-Co 313 bad')
		plt.plot(list(filedic[4][2].fiox)+list(filedic[4][4].fiox), list(filedic[4][2].tfy)+list(filedic[4][4].tfy), 'b-', label='(Fe Kalpha) Fe-Co 69 pure')
		plt.legend(loc='upper left',ncol=1)
		plt.show()
		#np.savetxt('C:/data/P01/Co_Kedge_XANES_TFY_Fe-Co313_Fe-Co69_FeCoKa.txt',(filedic[3][3].fiox,filedic[3][3].tfy,filedic[4][3].fiox,filedic[4][3].tfy))
		plt.plot(list(filedic[3][3].fiox)+list(filedic[3][5].fiox), list(filedic[3][3].tfy)+list(filedic[3][5].tfy), 'r-', label='(Co Kalpha) Fe-Co 313 bad')
		plt.plot(list(filedic[4][3].fiox)+list(filedic[4][5].fiox), list(filedic[4][3].tfy)+list(filedic[4][5].tfy), 'b-', label='(Co Kalpha) Fe-Co 69 pure')
		plt.legend(loc='upper left',ncol=1)
		plt.show()

	elif method == 11 and address < 7:
		#np.savetxt('C:/data/P01/Fe_Kedge_XANES_TFY_Fe-Co313_Fe-Co69_FeKb.txt',(list(filedic[5][2].fiox)+list(filedic[5][3].fiox), list(filedic[5][2].tfy)+list(filedic[5][3].tfy), list(filedic[6][2].fiox)+list(filedic[6][3].fiox), list(filedic[6][2].tfy)+list(filedic[6][3].tfy)))
		plt.plot(list(filedic[5][2].fiox)+list(filedic[5][3].fiox), list(filedic[5][2].tfy)+list(filedic[5][3].tfy), 'r-', label='(Fe Kbeta) Fe-Co 313 bad')
		plt.plot(list(filedic[6][2].fiox)+list(filedic[6][3].fiox), list(filedic[6][2].tfy)+list(filedic[6][3].tfy), 'b-', label='(Fe Kbeta) Fe-Co 69 pure')
		plt.legend(loc='upper left',ncol=1)
		plt.show()

	elif method == 12 and address < 3:
		plt.plot(filedic[0][2].fiox, filedic[0][2].i1, 'g-', label='(Co Kbeta) Co(III)py')
		plt.plot(filedic[1][2].fiox, filedic[1][2].i1, 'r-', label='(Co Kbeta) Fe-Co 313 bad')
		plt.plot(filedic[2][2].fiox, filedic[2][2].i1, 'b-', label='(Co Kbeta) Fe-Co 69 pure')
		plt.legend(loc='upper left',ncol=1)
		plt.show()

	elif method == 12 and address < 5:
		plt.plot(filedic[3][2].fiox, filedic[3][2].i1, 'r-', label='(Fe Kalpha) Fe-Co 313 bad')
		plt.plot(filedic[4][2].fiox, filedic[4][2].i1, 'b-', label='(Fe Kalpha) Fe-Co 69 pure')
		plt.legend(loc='upper left',ncol=1)
		plt.show()
		plt.plot(filedic[3][3].fiox, filedic[3][3].i1, 'r-', label='(Co Kalpha) Fe-Co 313 bad')
		plt.plot(filedic[4][3].fiox, filedic[4][3].i1, 'b-', label='(Co Kalpha) Fe-Co 69 pure')
		plt.legend(loc='upper left',ncol=1)
		plt.show()

	elif method == 12 and address < 7:
		plt.plot(list(filedic[5][2].fiox)+list(filedic[5][3].fiox), list(filedic[5][2].i1)+list(filedic[5][3].i1), 'r-', label='(Fe Kbeta) Fe-Co 313 bad')
		plt.plot(list(filedic[6][2].fiox)+list(filedic[6][3].fiox), list(filedic[6][2].i1)+list(filedic[6][3].i1), 'b-', label='(Fe Kbeta) Fe-Co 69 pure')
		plt.legend(loc='upper left',ncol=1)
		plt.show()

	elif method == 13:
		print("HERFD background...")
		for scan in filedic[address][2].bg:
			plt.plot(scan)

		plt.show()

		if address == 3 or address == 4:
			print("Co HERFD background...")
			for scan in filedic[address][3].bg:
				plt.plot(scan)

			plt.show()

	elif method == 14:
		#np.savetxt('C:/data/P01/Fe_Kedge_XANES_TFY_(Co_Kb)_Fe-Co313_Fe-Co69_(Fe_Ka)_Fe-Co69.txt',(list(filedic[1][3].fiox)+list(filedic[8][3].fiox), list(filedic[1][3].tfy)+list(filedic[8][3].tfy),list(filedic[2][3].fiox)+list(filedic[9][3].fiox), list(filedic[2][3].tfy)+list(filedic[9][3].tfy),list(filedic[4][2].fiox)+list(filedic[4][4].fiox), list(filedic[4][2].tfy)+list(filedic[4][4].tfy)))
		plt.plot(list(filedic[1][3].fiox)+list(filedic[8][3].fiox), list(filedic[1][3].tfy)+list(filedic[8][3].tfy), 'r-', label='(Co Kbeta-Fe TFY XAS) Fe-Co 313 bad')
		plt.plot(list(filedic[2][3].fiox)+list(filedic[9][3].fiox), list(filedic[2][3].tfy)+list(filedic[9][3].tfy), 'b-', label='(Co Kbeta-Fe TFY XAS) Fe-Co 69 pure')
		plt.plot(list(filedic[4][2].fiox)+list(filedic[4][4].fiox), list(filedic[4][2].tfy)+list(filedic[4][4].tfy), 'g-', label='(Fe Kalpha TFY) Fe-Co 69 pure')
		plt.legend(loc='upper left',ncol=1)
		plt.show()

	elif method == 15 and address != 9 and address != 13:
		plt.plot(filedic[address][0].pin3, filedic[address][0].i0, 'b', label='XES_pin3_vs._I0')
		plt.legend(loc='upper left',ncol=1)
		plt.show()
		plt.plot(filedic[address][0].pin3, filedic[address][0].i1, 'g', label='XES_pin3_vs._I1')
		plt.legend(loc='upper left',ncol=1)
		plt.show()
		plt.plot(filedic[address][0].pin3, filedic[address][0].tfy, 'r', label='XES_pin3_vs._TFY')
		plt.legend(loc='upper left',ncol=1)
		plt.show()

	elif method == 16 and address != 9 and address != 13:
		plt.plot(filedic[address][2].pin3, filedic[address][2].i0, 'b', label='HERFD_pin3_vs._I0')
		plt.legend(loc='upper left',ncol=1)
		plt.show()

gui()
