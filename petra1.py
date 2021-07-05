import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import os

filepath = 'C:/data/P01/raw/'

cudmpxes = filepath + 'CuI_dmp_ethanol_100mM_00722'
cudmpelastic = filepath + 'CuI_dmp_ethanol_100mM_elastic_00726'
cudmpxanes = filepath + 'CuI_dmp_ethanol_100mM_XAS_00724'
cudmpexafs = filepath + 'CuI_dmp_ethanol_100mM_XAS_00725'

cuixes = filepath + 'CuI_solid_00728'
cuielastic = filepath + 'CuI_solid_elastic-peak_00733'

cuso4xes = filepath + 'CuSO4_solid_00735'
cuso4elastic = filepath + 'CuSO4_solid_00736'
cuso4xanes = filepath + 'CuSO4_solid_XAS_00738'

cuac2xes = filepath + 'CuAc2_solid_00740'
cuac2xanes = filepath + 'CuAc2_solid_00741'

cuclxes = filepath + 'CuCl_solid_XES_00743'
cuclelastic = filepath + 'CuCl_solid_Elastic_00744'
cuclxanes = filepath + 'CuCl_solid_XAS_00745'

cucl2xes = filepath + 'CuCl2_solid_XES_00746'
cucl2elastic = filepath + 'CuCl2_solid_Elastic_00747'
cucl2xanes = filepath + 'CuCl2_solid_XAS_00748'
cucl2exafs = filepath + 'CuCl2_solid_XAS_00749'

cuoxes = filepath + 'CuO_solid_XES_00751' # too diluted
cuoelastic = filepath + 'CuO_solid_Elastic_00752'
cuoxanes = filepath + 'CuO_solid_XAS_00753'
cuoexafs = filepath + 'CuO_solid_XASnormalization_00754'

cu2oxes = filepath + 'Cu2O_solid_XES_00756'
cu2oelastic = filepath + 'Cu2O_solid_Elastic_00757'
cu2oxanes = filepath + 'Cu2O_solid_XAS_00758'
cu2oexafs = filepath + 'Cu2O_solid_XASnormalization_00759'

cufoilxes = filepath + 'Cu-foil_Kbeta_XES_00716'
cufoilelastic = filepath + 'Cu-foil_elastic-peak_00720'
cufoilxes2 = filepath + 'Cu-foil_Kbeta_XES_00717'

cobpyxes = filepath + 'CoII_bpy_100mM_H2O_XES_00580'
cobpyxanes = filepath + 'CoI_bpy_100mM_H2O_XES_00573'

conh3xes = filepath + 'CoIII_NH3_6_230mM_H2O_XES_00584'
conh3elastic = filepath + 'CoIII_NH3_6_230mM_H2O_XES_00587'
conh3xanes = filepath + 'CoIII_NH3_6_230mM_H2O_XES_00583'

cosepxes = filepath + 'CoIII_sep_3_21mM_H2O_XAS_00598'
cosepxanes = filepath + 'CoIII_sep_3_21mM_H2O_XAS_00593'
cosepxes2 = filepath + 'CoIII_sep_3_21mM_H2O_XAS_00599'

coc2o4xes = filepath + 'XES_Kalpha_CoSQ_C2O4_00699'
coc2o4elastic = filepath + 'CoSQ_C2O4_00710'
coc2o4xanes = filepath + 'CoSQ_C2O4_00709'

cofoilelastic = filepath + 'Co-foil_elastic_00562'
cofoilxanes = filepath + 'Co-foil_Kalpha-XAS_00559'

elines = [[57, 82, 110, 139, 169], [61, 86, 114, 142, 172], [69, 94, 122, 151, 181], [67, 92, 120, 150, 179], [43, 68, 96, 124, 154], [59, 84, 112, 142, 172], [68, 93, 121, 150, 180], [51, 77, 104, 133, 164], [48, 73, 101, 130, 159], [22, 41, 54, 69, 92, 110, 175], [22, 41, 54, 70, 92, 110, 175], [22, 41, 54, 70, 92, 110, 175], [12, 59, 80, 99, 164], [23, 42, 55, 71, 92, 111, 176]]
c2o4elastic = [6, 54, 74, 93, 157]
dataset = [[cudmpxes,cudmpelastic,cudmpxanes,cudmpexafs],[cuixes,cuielastic],[cuso4xes,cuso4elastic,cuso4xanes],[cuac2xes,None,cuac2xanes],[cuclxes,cuclelastic,cuclxanes],[cucl2xes,cucl2elastic,cucl2xanes,cucl2exafs],[cuoxes,cuoelastic,cuoxanes,cuoexafs],[cu2oxes,cu2oelastic,cu2oxanes,cu2oexafs],[cufoilxes,cufoilelastic,cufoilxes2],[cobpyxes,None,cobpyxanes],[conh3xes,conh3elastic,conh3xanes],[cosepxes,None,cosepxanes,cosepxes2],[coc2o4xes,coc2o4elastic,coc2o4xanes],[None,cofoilelastic,cofoilxanes]]
filedic = {}


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

		self.fset = np.loadtxt(filefio, skiprows = 121, usecols = (3,4,5,6)).transpose()
		self.fiox = self.fset[0] # energy axis (mono) of the measurement = # of files
		#self.i0 = self.fset[2] # we didn't have i0 monitor...shit
		self.pin3 = self.fset[3] # maybe pin3 was i0 monitor
		test_i1 = 1e6-self.fset[2] # transmission mode, pin2
		#test_tfy = self.fset[1]/self.i0 # we didn't have tfy
		'''
		if sample == 12:
			plt.plot(self.fset[3],self.fset[2],'b')
			plt.show()
		'''
		self.i1 = test_i1/max(test_i1) # Normalization?
		#self.tfy = test_tfy/max(test_tfy)

		self.bg = [None]*len(filedir) # mostly electronic noise or background scattered photon along x-axis

		for j, scan in enumerate(temp_dset):
			if sample == 0:
				self.bg[j] = np.average(list(scan[5:40])+list(scan[64:73])+list(scan[92:101])+list(scan[119:131])+list(scan[149:162])+list(scan[185:-5]), axis=0) # background noise
				self.dset[j] = scan - self.bg[j] # subtract
			elif sample == 4:
				self.bg[j] = np.average(list(scan[5:30])+list(scan[50:58])+list(scan[80:85])+list(scan[110:115])+list(scan[140:145])+list(scan[175:-5]), axis=0) # background noise
				self.dset[j] = scan - self.bg[j] # subtract
			elif sample == 6:
				self.bg[j] = np.average(list(scan[5:59])+list(scan[76:85])+list(scan[103:113])+list(scan[131:144])+list(scan[160:174]), axis=0) # background noise
				self.dset[j] = scan - self.bg[j] # subtract
			elif sample < 9:
				self.bg[j] = np.average(scan[5:30], axis=0) # background noise
				self.dset[j] = scan - self.bg[j] # subtract
			elif sample != 12:
				self.bg[j] = np.average(list(scan[5:15])+list(scan[30:35])+list(scan[75:85])+list(scan[120:150])+list(scan[160:169])+list(scan[185:-5]), axis=0) # background noise
				self.dset[j] = scan - self.bg[j] # subtract
			else:
				self.bg[j] = np.average(list(scan[110:135])+list(scan[175:-5]), axis=0) # background noise
				self.dset[j] = scan - self.bg[j] # subtract
		'''
		if sample == 12:
			plt.imshow(self.dset[-1], cmap='terrain')
			plt.show()
		'''
		filedic[sample].append('dset_nbg')

	def elastic(self, sample, ka): # FWHM of Si(111) mono ~1 eV
		ey = elines[sample]
		if sample < 9 or sample == 13:
			emity = [(num-5) + [max(a) for a in self.dset[30][num-5:num+5]].index(max([max(a) for a in self.dset[30][num-5:num+5]])) for num in ey] # max y within each emission line
		elif sample != 12:
			emity = [(num-5) + [max(a) for a in self.dset[4][num-5:num+5]].index(max([max(a) for a in self.dset[4][num-5:num+5]])) for num in ey] # max y within each emission line
		else:
			emity = [(num-5) + [max(a) for a in self.dset[0][num-5:num+5]].index(max([max(a) for a in self.dset[0][num-5:num+5]])) for num in c2o4elastic] # max y within each emission line
		'''
		if sample > 8:
			for spec in self.dset[:10]:
				plt.imshow(spec, cmap='terrain')
				plt.show()
		'''
		if ka == 0:
			# 2d array of signal x position for each elastic scan
			emitx = [[list(scan[numb]).index(max(list(scan[numb]))) for scan in self.dset[10:50]] for numb in emity]

		else:
			# Co Kalpha 2d array of signal x position
			emitx = [[list(scan[numb]).index(max(list(scan[numb]))) for scan in self.dset[:10]] for numb in emity]

		self.eset = [None]*len(emity) # linear fit of each emission line emitx to fio file, and calibrated x-axis
		for t, pixel in enumerate(emitx):
			linearfit = np.vstack([pixel, np.ones(len(pixel))]).T
			if ka == 0:
				o, d = np.linalg.lstsq(linearfit, self.fiox[10:50])[0]
			else:
				o, d = np.linalg.lstsq(linearfit, self.fiox[:10])[0]

			self.eset[t] = [n*o + d for n in range(487)]

		filedic[sample].append('elastic_calibration')

		self.xesx = self.eset[2] # standard x-axis; 3rd Xtal as a reference
		'''
		if sample == 0:
			plt.plot(self.eset[0], np.average(self.dset[10][emity[0]-3:emity[0]+3], axis=0))
			plt.plot(self.eset[0], np.average(self.dset[30][emity[0]-3:emity[0]+3], axis=0))
			plt.plot(self.eset[0], np.average(self.dset[50][emity[0]-3:emity[0]+3], axis=0))
			plt.plot(self.eset[0], np.average(self.dset[70][emity[0]-3:emity[0]+3], axis=0))
			plt.plot(self.eset[0], np.average(self.dset[90][emity[0]-3:emity[0]+3], axis=0))
			plt.show()
		'''

	def xes(self, sample, ka):
		ey = elines[sample]
		emity = [(num-5) + [max(a) for a in self.dset[1][num-5:num+5]].index(max([max(a) for a in self.dset[1][num-5:num+5]])) for num in ey] # max y within each emission line

		self.emitset = np.average([[np.average(scan[numb-8:numb+8], axis=0) for scan in self.dset] for numb in emity], axis=1) # averaged emission spectra from each crystal
		'''
		if sample > 10:
			for spec in self.emitset:
				plt.plot(spec)

			plt.show()
		'''
		if ka == 0:
			xesy = self.emitset[2] # standard y-axis; this Xtal must be used later for x-axis
			for z, scan2 in enumerate(self.emitset):
				if z != 2:
					offset = list(self.emitset[2]).index(max(list(self.emitset[2]))) - list(scan2).index(max(list(scan2)))
					if offset > 0:
						temp3 = np.concatenate((xesy[:offset], (scan2[:-offset]+xesy[offset:])/2))

						#plt.plot(scan2[:-offset], 'b')
						#plt.plot(self.emitset[2][offset:], 'r')
						#plt.show()

					elif offset < 0:
						temp3 = np.concatenate(((xesy[:offset]+scan2[-offset:])/2, xesy[offset:]))

						#plt.plot(scan2[-offset:], 'b')
						#plt.plot(self.emitset[2][:offset], 'r')
						#plt.show()

					else:
						temp3 = (xesy+scan2)/2

						#plt.plot(scan2, 'b')
						#plt.plot(self.emitset[2], 'r')
						#plt.show()

					xesy = temp3 # all emission lines from each crystal overlapped

			self.xesy=xesy#/np.trapz(xesy) # spectrum area normalization

		filedic[sample].append('xes_areanorm')

	def herfd2d(self, sample, ka):
		ey = elines[sample]
		emity = [(num-5) + [max(a) for a in self.dset[-5][num-5:num+5]].index(max([max(a) for a in self.dset[-5][num-5:num+5]])) for num in ey] # max y within each emission line
		'''
		if sample > 8:
			plt.imshow(self.dset[-5], cmap='terrain')
			plt.show()
		'''
		self.herfdset = [None]*len(self.dset)
		#print(self.fiox[40])

		offset = [list(self.dset[-5][emity[2]]).index(max(list(self.dset[-5][emity[2]]))) - list(self.dset[-5][e]).index(max(list(self.dset[-5][e]))) for e in emity]
		#print(offset)

		for y, scan in enumerate(self.dset):
			temp2 = [np.average(scan[numb-8:numb+8], axis=0) for numb in emity] # each emission line will be overlapped and then added into HERFD 2d image

			if ka == 0:
				herfdset = temp2[2] # standard y-axis; this Xtal must be used later for x-axis
				for c, scan2 in enumerate(temp2):
					if c != 2:
						if offset[c] > 0:
							temp3 = np.concatenate((herfdset[:offset[c]], (scan2[:-offset[c]]+herfdset[offset[c]:])/2))

						elif offset[c] < 0:
							temp3 = np.concatenate(((herfdset[:offset[c]]+scan2[-offset[c]:])/2, herfdset[offset[c]:]))

						else:
							temp3 = (herfdset+scan2)/2

						herfdset = temp3

				self.herfdset[y]=herfdset*10e3#/self.i0[y] # i0 normalization
		'''
		if sample == 12:
			plt.imshow(self.herfdset, cmap='terrain', aspect="auto")
			plt.show()
		'''
		filedic[sample].append('herfd2d_i0norm')


def gui():
	print("Hi, welcome to PILATUS 100k data analysis tool.")
	print("\nCurrently we have those items:\n(Dec 2017) Cu Kbeta\n0. Cu(I)dmp_Ethanol\n1. Cu(I)I_BN\n2. Cu(II)SO4_BN\n3. Cu(II)(acac)2_BN\n4. Cu(I)Cl_BN\n5. Cu(II)Cl2_BN\n6. Cu(II)O_BN\n7. Cu(I)2O_BN\n8. Cu_foil\n\n(Dec 2017) Co Kalpha\n9. Co(II)bpy_Water\n10. Co(III)NH3_Water\n11. Co(III)sep3_Water\n12. Co(II)C2O4_Cellulose\n13. Co_foil\nStart analysis...\n")
	sample = 0
	while sample < len(dataset):
		filedic[sample] = ['xes', 'elastic', 'xanes', 'exafs']

		if sample != 13:
			filedic[sample][0] = getdata(sample, dataset[sample][0])
			filedic[sample][0].xes(sample, 0)
			print("{} sample XES loaded.".format(str(sample)))

		if sample != 3 and sample != 9 and sample != 10 and sample != 11 and sample != 12:
			filedic[sample][1] = getdata(sample, dataset[sample][1])
			filedic[sample][1].elastic(sample, 0)
			print("{} sample elastic scan calibrated.".format(str(sample)))
		elif sample == 10 or sample == 12:
			filedic[sample][1] = getdata(sample, dataset[sample][1])
			filedic[sample][1].elastic(sample, 1)
			print("{} sample elastic scan calibrated.".format(str(sample)))

		if sample != 1 and sample != 8:
			filedic[sample][2] = getdata(sample, dataset[sample][2])
			filedic[sample][2].herfd2d(sample, 0)
			print("{} sample HERFD XANES loaded.".format(str(sample)))
		elif sample == 8:
			filedic[sample][2] = getdata(sample, dataset[sample][2])
			filedic[sample][2].xes(sample, 0)
			print("{} sample XES_2 loaded.".format(str(sample)))

		if sample != 1 and sample != 2 and sample != 3 and sample != 4 and sample < 8:
			filedic[sample][3] = getdata(sample, dataset[sample][3])
			filedic[sample][3].herfd2d(sample, 0)
			print("{} sample HERFD EXAFS loaded.".format(str(sample)))
		elif sample == 11:
			filedic[sample][3] = getdata(sample, dataset[sample][3])
			filedic[sample][3].xes(sample, 0)
			print("{} sample XES_2 loaded.".format(str(sample)))

		sample += 1

	i = 0
	while i < 1:
		opt = int(input("\nPlease select an option\n0. Current datalist\n1. Plotting data\n2. Exit\n"))
		if opt == 0:
			print("\nWe have currently this datalist.")
			print(filedic)
		elif opt == 1:
			plot = [int(x) for x in input("\nPlease input sample# and plot method\n0. XES\n1. XES_all_images\n2. XES_I1\n3. XES_background\n4. XES_each_crystal\n5. Elastic_scan_images\n6. HERFD\n7. HERFD_all_images\n8. HERFD_I1\n9. HERFD_background\n").split()]
			plotting(plot[0], plot[1])
		else: i = 1


def plotting(address, method):
	if method == 0 and address < 9:
		#np.savetxt('C:/data/P01/CuCo/CuKb_Cudmp.txt',(filedic[0][1].xesx, ((filedic[0][0].xesy-np.average(filedic[0][1].dset[1],axis=0))-(filedic[0][0].xesy-np.average(filedic[0][1].dset[1],axis=0))[0])/np.trapz(((filedic[0][0].xesy-np.average(filedic[0][1].dset[1],axis=0))-(filedic[0][0].xesy-np.average(filedic[0][1].dset[1],axis=0))[0]))*(0.3157/0.2851)))
		#np.savetxt('C:/data/P01/CuCo/CuKb_CuI.txt',(filedic[1][1].xesx, ((filedic[1][0].xesy-np.average(filedic[1][1].dset[1],axis=0))-(filedic[1][0].xesy-np.average(filedic[1][1].dset[1],axis=0))[0])/np.trapz(((filedic[1][0].xesy-np.average(filedic[1][1].dset[1],axis=0))-(filedic[1][0].xesy-np.average(filedic[1][1].dset[1],axis=0))[0]))))
		#np.savetxt('C:/data/P01/CuCo/CuKb_CuCl.txt',(filedic[4][1].xesx, ((filedic[4][0].xesy-np.average(filedic[4][1].dset[1],axis=0))-(filedic[4][0].xesy-np.average(filedic[4][1].dset[1],axis=0))[0])/np.trapz(((filedic[4][0].xesy-np.average(filedic[4][1].dset[1],axis=0))-(filedic[4][0].xesy-np.average(filedic[4][1].dset[1],axis=0))[0]))*(0.3157/0.2257)))
		#np.savetxt('C:/data/P01/CuCo/CuKb_Cu2O.txt',(filedic[7][1].xesx, ((filedic[7][0].xesy-np.average(filedic[7][1].dset[1],axis=0))-(filedic[7][0].xesy-np.average(filedic[7][1].dset[1],axis=0))[0])/np.trapz(((filedic[7][0].xesy-np.average(filedic[7][1].dset[1],axis=0))-(filedic[7][0].xesy-np.average(filedic[7][1].dset[1],axis=0))[0]))*(0.3157/0.3069)))
		#plt.plot(filedic[8][1].xesx, (((filedic[8][0].xesy+filedic[8][2].xesy)/2-np.average(filedic[8][1].dset[1],axis=0))-((filedic[8][0].xesy+filedic[8][2].xesy)/2-np.average(filedic[8][1].dset[1],axis=0))[0])/np.trapz((((filedic[8][0].xesy+filedic[8][2].xesy)/2-np.average(filedic[8][1].dset[1],axis=0))-((filedic[8][0].xesy+filedic[8][2].xesy)/2-np.average(filedic[8][1].dset[1],axis=0))[0]))*(0.3157/0.1688), label='Cu_foil')
		plt.plot(filedic[0][1].xesx, ((filedic[0][0].xesy-np.average(filedic[0][1].dset[1],axis=0))-(filedic[0][0].xesy-np.average(filedic[0][1].dset[1],axis=0))[0])/np.trapz(((filedic[0][0].xesy-np.average(filedic[0][1].dset[1],axis=0))-(filedic[0][0].xesy-np.average(filedic[0][1].dset[1],axis=0))[0]))*(0.3157/0.2851), label='Cu(I)dmp_Ethanol')
		plt.plot(filedic[1][1].xesx, ((filedic[1][0].xesy-np.average(filedic[1][1].dset[1],axis=0))-(filedic[1][0].xesy-np.average(filedic[1][1].dset[1],axis=0))[0])/np.trapz(((filedic[1][0].xesy-np.average(filedic[1][1].dset[1],axis=0))-(filedic[1][0].xesy-np.average(filedic[1][1].dset[1],axis=0))[0])), label='Cu(I)I_BN')
		plt.plot(filedic[4][1].xesx, ((filedic[4][0].xesy-np.average(filedic[4][1].dset[1],axis=0))-(filedic[4][0].xesy-np.average(filedic[4][1].dset[1],axis=0))[0])/np.trapz(((filedic[4][0].xesy-np.average(filedic[4][1].dset[1],axis=0))-(filedic[4][0].xesy-np.average(filedic[4][1].dset[1],axis=0))[0]))*(0.3157/0.2257), label='Cu(I)Cl_BN')
		plt.plot(filedic[7][1].xesx, ((filedic[7][0].xesy-np.average(filedic[7][1].dset[1],axis=0))-(filedic[7][0].xesy-np.average(filedic[7][1].dset[1],axis=0))[0])/np.trapz(((filedic[7][0].xesy-np.average(filedic[7][1].dset[1],axis=0))-(filedic[7][0].xesy-np.average(filedic[7][1].dset[1],axis=0))[0]))*(0.3157/0.3069), label='Cu(I)2O_BN')
		plt.legend(loc='upper right',ncol=1)
		plt.show()

		#np.savetxt('C:/data/P01/CuCo/CuKb_CuSO4.txt',(filedic[2][1].xesx, ((filedic[2][0].xesy-np.average(filedic[2][1].dset[1],axis=0))-(filedic[2][0].xesy-np.average(filedic[2][1].dset[1],axis=0))[0])/np.trapz(((filedic[2][0].xesy-np.average(filedic[2][1].dset[1],axis=0))-(filedic[2][0].xesy-np.average(filedic[2][1].dset[1],axis=0))[0]))*(0.3066/0.2973)))
		#np.savetxt('C:/data/P01/CuCo/CuKb_Cu(acac)2.txt',(filedic[2][1].xesx, ((filedic[3][0].xesy-np.average(filedic[2][1].dset[1],axis=0))-(filedic[3][0].xesy-np.average(filedic[2][1].dset[1],axis=0))[0])/np.trapz(((filedic[3][0].xesy-np.average(filedic[2][1].dset[1],axis=0))-(filedic[3][0].xesy-np.average(filedic[2][1].dset[1],axis=0))[0]))))
		#np.savetxt('C:/data/P01/CuCo/CuKb_CuCl2.txt',(filedic[5][1].xesx, ((filedic[5][0].xesy-np.average(filedic[5][1].dset[1],axis=0))-(filedic[5][0].xesy-np.average(filedic[5][1].dset[1],axis=0))[0])/np.trapz(((filedic[5][0].xesy-np.average(filedic[5][1].dset[1],axis=0))-(filedic[5][0].xesy-np.average(filedic[5][1].dset[1],axis=0))[0]))*(0.3066/0.2808)))
		#np.savetxt('C:/data/P01/CuCo/CuKb_CuO.txt',(filedic[6][1].xesx, ((filedic[6][0].xesy-np.average(filedic[6][1].dset[1],axis=0))-(filedic[6][0].xesy-np.average(filedic[6][1].dset[1],axis=0))[0])/np.trapz(((filedic[6][0].xesy-np.average(filedic[6][1].dset[1],axis=0))-(filedic[6][0].xesy-np.average(filedic[6][1].dset[1],axis=0))[0]))*(0.3066/0.2662)))
		#plt.plot(filedic[8][1].xesx, (((filedic[8][0].xesy+filedic[8][2].xesy)/2-np.average(filedic[8][1].dset[1],axis=0))-((filedic[8][0].xesy+filedic[8][2].xesy)/2-np.average(filedic[8][1].dset[1],axis=0))[0])/np.trapz((((filedic[8][0].xesy+filedic[8][2].xesy)/2-np.average(filedic[8][1].dset[1],axis=0))-((filedic[8][0].xesy+filedic[8][2].xesy)/2-np.average(filedic[8][1].dset[1],axis=0))[0]))*(0.3157/0.18), label='Cu_foil')
		plt.plot(filedic[2][1].xesx, ((filedic[2][0].xesy-np.average(filedic[2][1].dset[1],axis=0))-(filedic[2][0].xesy-np.average(filedic[2][1].dset[1],axis=0))[0])/np.trapz(((filedic[2][0].xesy-np.average(filedic[2][1].dset[1],axis=0))-(filedic[2][0].xesy-np.average(filedic[2][1].dset[1],axis=0))[0]))*(0.3066/0.2973), label='Cu(II)SO4_BN')
		plt.plot(filedic[2][1].xesx, ((filedic[3][0].xesy-np.average(filedic[2][1].dset[1],axis=0))-(filedic[3][0].xesy-np.average(filedic[2][1].dset[1],axis=0))[0])/np.trapz(((filedic[3][0].xesy-np.average(filedic[2][1].dset[1],axis=0))-(filedic[3][0].xesy-np.average(filedic[2][1].dset[1],axis=0))[0])), label='Cu(II)(acac)2_BN')
		plt.plot(filedic[5][1].xesx, ((filedic[5][0].xesy-np.average(filedic[5][1].dset[1],axis=0))-(filedic[5][0].xesy-np.average(filedic[5][1].dset[1],axis=0))[0])/np.trapz(((filedic[5][0].xesy-np.average(filedic[5][1].dset[1],axis=0))-(filedic[5][0].xesy-np.average(filedic[5][1].dset[1],axis=0))[0]))*(0.3066/0.2808), label='Cu(II)Cl2_BN')
		plt.plot(filedic[6][1].xesx, ((filedic[6][0].xesy-np.average(filedic[6][1].dset[1],axis=0))-(filedic[6][0].xesy-np.average(filedic[6][1].dset[1],axis=0))[0])/np.trapz(((filedic[6][0].xesy-np.average(filedic[6][1].dset[1],axis=0))-(filedic[6][0].xesy-np.average(filedic[6][1].dset[1],axis=0))[0]))*(0.3066/0.2662), label='Cu(II)O_BN')
		plt.legend(loc='upper right',ncol=1)
		plt.show()

	elif method == 0 and address > 8:
		#np.savetxt('C:/data/P01/CuCo/CoKa_Co2+bpy.txt',(filedic[10][1].xesx, (filedic[9][0].xesy-np.average(filedic[10][1].dset[30],axis=0))/np.trapz((filedic[9][0].xesy-np.average(filedic[10][1].dset[30],axis=0)))))
		#np.savetxt('C:/data/P01/CuCo/CoKa_Co3+NH3.txt',(filedic[10][1].xesx, (filedic[10][0].xesy-np.average(filedic[10][1].dset[30],axis=0))/np.trapz((filedic[10][0].xesy-np.average(filedic[10][1].dset[30],axis=0)))*(0.4024/0.4619)))
		#np.savetxt('C:/data/P01/CuCo/CoKa_Co3+sep3.txt',(filedic[10][1].xesx, ((filedic[11][0].xesy+filedic[11][3].xesy)/2-np.average(filedic[10][1].dset[30],axis=0))/np.trapz(((filedic[11][0].xesy+filedic[11][3].xesy)/2-np.average(filedic[10][1].dset[30],axis=0)))))
		#np.savetxt('C:/data/P01/CuCo/CoKa_Co2+C2O4.txt',(filedic[12][1].xesx, (filedic[12][0].xesy-np.average(filedic[12][1].dset[30],axis=0))/np.trapz((filedic[12][0].xesy-np.average(filedic[12][1].dset[30],axis=0)))))
		plt.plot(filedic[10][1].xesx, (filedic[9][0].xesy-np.average(filedic[10][1].dset[30],axis=0))/np.trapz((filedic[9][0].xesy-np.average(filedic[10][1].dset[30],axis=0))), label='Co(II)bpy_Water')
		plt.plot(filedic[10][1].xesx, (filedic[10][0].xesy-np.average(filedic[10][1].dset[30],axis=0))/np.trapz((filedic[10][0].xesy-np.average(filedic[10][1].dset[30],axis=0)))*(0.4024/0.4619), label='Co(III)NH3_Water')
		plt.plot(filedic[10][1].xesx, ((filedic[11][0].xesy+filedic[11][3].xesy)/2-np.average(filedic[10][1].dset[30],axis=0))/np.trapz(((filedic[11][0].xesy+filedic[11][3].xesy)/2-np.average(filedic[10][1].dset[30],axis=0))), label='Co(III)sep3_Water')
		plt.plot(filedic[12][1].xesx, (filedic[12][0].xesy-np.average(filedic[12][1].dset[30],axis=0))/np.trapz((filedic[12][0].xesy-np.average(filedic[12][1].dset[30],axis=0))), label='Co(II)C2O4_Cellulose')
		plt.legend(loc='upper right',ncol=1)
		plt.show()

	elif method == 1 and address != 13:
		for image in filedic[address][0].dset:
			plt.imshow(image)
			plt.show()

	elif method == 2 and address != 13:
		plt.plot(filedic[address][0].i1, 'r-')
		plt.show()

	elif method == 3 and address != 13:
		for scan in filedic[address][0].bg:
			plt.plot(scan)

		plt.show()

	elif method == 4 and address != 3 and address != 13 and address != 9 and address != 11:
		for n, scan in enumerate(filedic[address][0].emitset):
			plt.plot(filedic[address][1].eset[n], scan)

		plt.show()

	elif method == 5 and address != 3 and address != 9 and address != 11:
		for image in filedic[address][1].dset:
			plt.imshow(image)
			plt.show()

	elif method == 6 and address != 1 and address != 8:
		print("Enter x-axis range to integrate...\n")
		#np.savetxt('C:/data/P01/CuCo/CoKa_HERFD2d_Co3(NH3)6_H2O.txt',(filedic[address][2].herfdset))
		#np.savetxt('C:/data/P01/CuCo/CoKa_HERFD2d_Co3(NH3)6_H2O_yaxis.txt',(filedic[address][2].fiox))
		#np.savetxt('C:/data/P01/CuCo/CoKa_HERFD2d_Co3(NH3)6_H2O_xaxis.txt',(filedic[10][1].xesx))
		plt.imshow(filedic[address][2].herfdset, cmap='terrain', aspect="auto")
		plt.show()
		#plt.imshow(np.array(filedic[address][2].herfdset[15:])/max(filedic[address][2].herfdset[69])-np.array(filedic[address+10][2].herfdset[15:])/max(filedic[address+10][2].herfdset[69]), cmap='terrain', aspect="auto")
		#plt.show()
		if str(input("Do you want to integrate?(y/n) ")) == 'y':
			enterx=[int(v) for v in input("Where to where? ").split()]
			plt.plot(filedic[address][2].fiox, np.average(np.transpose(filedic[address][2].herfdset)[enterx[0]:enterx[1]], axis=0), 'b')
			#np.savetxt('C:/data/P01/CuCo/Co(III)sep3_H2O_XANES.txt',(filedic[address][2].fiox, np.average(np.transpose(filedic[address][2].herfdset)[enterx[0]:enterx[1]], axis=0)))
			#plt.plot(filedic[1][2].fiox, filedic[1][2].herfdy, 'b-', label='(Co Kbeta) Fe-Co 313')
			#plt.plot(filedic[2][2].fiox, filedic[2][2].herfdy, 'g-', label='(Co Kbeta) Fe-Co 69')
			plt.show()

	elif method == 7 and address != 1 and address != 8:
		for scan in filedic[address][2].dset:
			plt.imshow(scan, cmap='terrain')
			plt.show()

	elif method == 8 and address != 1 and address != 8:
		print("HERFD I1...")
		plt.plot(filedic[address][2].fiox, filedic[address][2].i1, 'r-')
		plt.show()

	elif method == 9 and address != 1 and address != 8:
		print("HERFD background...")
		for scan in filedic[address][2].bg:
			plt.plot(scan)

		plt.show()


gui()
