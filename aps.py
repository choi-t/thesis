import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.special import erf
import os
import csv

# Choose data address interested in
phen3 = 'C:/data/aps/phen3'
phen2dppz = 'C:/data/aps/phen2dppz'
tap3 = 'C:/data/aps/tap3'
tap2dppz = 'C:/data/aps/tap2dppz'
phen3w = 'C:/data/aps/phen3w'
phen2dppzw = 'C:/data/aps/phen2dppzw'
tap2dppzw = 'C:/data/aps/tap2dppzw'
feff = 'C:/data/aps/feff.txt'
saved = 'C:/data/aps/test.txt'

dphen2dppz = 'C:/data/aps/delay/phen2dppz'
dtap3 = 'C:/data/aps/delay/tap3'
dtap2dppz = 'C:/data/aps/delay/tap2dppz'
dphen3w = 'C:/data/aps/delay/phen3w'
dphen2dppzw = 'C:/data/aps/delay/phen2dppzw'
dtap2dppzw = 'C:/data/aps/delay/tap2dppz'

dataset = [phen3, phen2dppz, tap3, tap2dppz, phen3w, phen2dppzw, tap2dppzw]
delayset = [dphen2dppz, dtap3, dtap2dppz, dphen3w, dphen2dppzw, dtap2dppzw]
decay = [19, 24, 10, 13, 9, 9]
slice = [[0,20], [0,34], [0,36], [0,12], [0,3], [0,3], [0,3]]
filedic = {}
delaydic = {}

# column[1] = LP, column[2] = UP, column[3] = T
def findmax(x, gs, es):
	n1gs = np.argmax(gs[:48])
	n2gs = np.argmax(gs[:60])
	n1es = np.argmax(es[:50])
	n2es = np.argmax(es[:65])
	
	return	(((x[n2es]+x[n1es])/2)-((x[n2gs]+x[n1gs])/2))*1000

def edgee(xs, ygs, yes):
	target = 0.35
	tarray = np.empty(3, '14f4')
	i = 0
	while target < 1.01:
		tarray[0][i] = target
		tarray[1][i] = xs[np.argmin(abs(ygs[:45]-target))]
		tarray[2][i] = xs[np.argmin(abs(yes[:50]-target))]
		i += 1
		target += 0.05
	
	temp = tarray[2] - tarray[1]
	avrg = np.sum(temp)/len(temp)

	return	avrg*1000

def findderv(x, gs, es):
	tw = np.empty(2, '202f4')
	l = 0
	while l < (len(x)-1):
		tw[0][l] = (gs[l+1]-gs[l])/(x[l+1]-x[l])
		tw[1][l] = (es[l+1]-es[l])/(x[l+1]-x[l])
		l += 1

	return ((x[np.argmax(tw[1][40:])+40])-(x[np.argmax(tw[0][40:])+40]))*1000, tw
	
def func(x, a, b, t):
	return a*np.exp(-x/t) + b
	
class getdata:
	def __init__(self, sample):
		files = os.listdir(dataset[sample])
		if sample == 4:
			self.dset = np.empty((len(files), 5), '86f4')
		elif sample == 5:
			self.dset = np.empty((len(files), 5), '91f4')
		elif sample == 6:
			self.dset = np.empty((len(files), 5), '75f4')
		else:
			self.dset = np.empty((len(files), 5), '203f4')
		
		for num, file in enumerate(files):

			fullpath = os.path.join(dataset[sample], file)
			mytextfile = np.loadtxt(fullpath, skiprows = 1)	
			mytextfile = list(zip(*mytextfile))
	
			self.dset[num, 0, :] = mytextfile[0]
			self.dset[num, 1, :] = mytextfile[1]
			self.dset[num, 2, :] = mytextfile[2]
			self.dset[num, 3, :] = mytextfile[3]
			self.dset[num, 4, :] = mytextfile[4]
		
		self.avrg = np.sum(self.dset, axis = 0)/len(files)
		filedic[sample].append('dset')
	
	def errorcheck(self, sample):
		index = np.empty(len(self.dset), '5f4')
		for i, scan in enumerate(self.dset):
			index[i][0] = i+1
			index[i][1] = scan[1][45] - self.avrg[1][45]
			index[i][2] = scan[2][45] - self.avrg[2][45]
			index[i][3] = scan[1][70] - self.avrg[1][70]
			index[i][4] = scan[2][70] - self.avrg[2][70]
		self.index = list(zip(*index))

		self.rset = self.dset[slice[sample][0]:slice[sample][1]]
		self.ravg = np.sum(self.rset, axis = 0)/len(self.rset)
		filedic[sample].append('corrected')
		
	def excited(self, sample):
		self.x = range(1,80,1)
		self.eset = [None]*79
		self.peak = [None]*79
		self.drv = [None]*79
		self.wdrv = np.empty((79, 2), '202f4')
		for j, n in enumerate(self.x):
			esmodel = self.ravg[3]*(100/n) + self.ravg[2]
			self.eset[j] = edgee(self.ravg[0], self.ravg[2], esmodel)
			self.peak[j] = findmax(self.ravg[0], self.ravg[2], esmodel)
			self.drv[j], self.wdrv[j] = findderv(self.ravg[0], self.ravg[2], esmodel)
		
		filedic[sample].append('excited')
				
	def exafs(self, sample):
		tempx = self.ravg[0][41:]*1000
		tempy = self.ravg[2][41:]
		tempk = ((tempx - (tempx[0]))**(0.5))*0.51232
		
		spl = UnivariateSpline(tempx, tempy, k=5)
		#chik = chiy - spl(tempk)
		#window = np.hanning(len(chik))
		#for p, num in enumerate(chik):
		#chik[p] = num*(tempk[p]**2)*window[p]
		
		#chir = chik[35:]
		
		#test = np.fft.fft(chik)
		
		#plt.plot(tempx, tempy, 'r-', tempx, spl(tempx), 'b-')
		#plt.show()

class delay:
	def __init__(self, sample):
		files = os.listdir(delayset[sample])
		if sample > 1:
			dset = np.empty((len(files), 2), '51f8')
		else:	dset = np.empty((len(files), 2), '61f8')
		
		for num, file in enumerate(files):

			fullpath = os.path.join(delayset[sample], file)
			mytextfile = np.loadtxt(fullpath)
			mytextfile = list(zip(*mytextfile))
	
			dset[num, 0, :] = mytextfile[0]
			dset[num, 1, :] = mytextfile[1]
		
		self.dscan = np.sum(dset, axis = 0)/len(files)
		self.dscan[0] = self.dscan[0]*(-0.001)
		delaydic[sample].append('dset')
		
		self.xfit = self.dscan[0][decay[sample]:]
		popt, pcov = curve_fit(func, self.xfit, self.dscan[1][decay[sample]:], p0=(-0.01, -0.001, 100))
		self.decayf = func(self.xfit, *popt)
		self.popt = popt
		delaydic[sample].append('monoexponential fit')
		
def fft(self):
	ppeak = np.arange(-0.007, 0.008, 0.001)
	self.ppx = np.arange(22.114, 22.129, 0.001)[:15]
	self.ppy = 0.39*np.cos(2*np.pi*35.7142857*ppeak) #cos intensity fitting needed

	edge = simps(self.nobg[32:72], self.avrg[0][32:72]) #check if x-range correct or not
	peakarea = simps(self.ppy, self.ppx)
	self.ion = (edge-peakarea)/peakarea
	
def plotting(address, method, sample):
	if method == 0:
		for scan in address.dset:
			plt.plot(scan[0], scan[4], 'b-')
		plt.show()
	elif method == 1:
		for scan in address.dset:
			plt.plot(scan[0], scan[1], 'b--')
			plt.plot(scan[0], scan[2], 'r--')
		plt.show()
	elif method == 2:
		plt.plot(address.index[0], address.index[1], 'r-', address.index[0], address.index[2], 'b-', address.index[0], address.index[3], 'r--', address.index[0], address.index[4], 'b--')
		plt.show()
	elif method == 3:
		fig, ax1 = plt.subplots()
		ax1.plot(address.ravg[0], address.ravg[1], 'g-', address.ravg[0], address.ravg[2], 'b-')
		ax1.set_xlabel('Energy (keV)')
		ax1.set_ylabel('Fluorescence Yield (a.u.)', color='b')
		ax1.tick_params('y', colors = 'b')

		ax2 = ax1.twinx()
		ax2.plot(address.ravg[0], address.ravg[3], 'r-')
		ax2.set_ylabel('Transient (a.u.)', color = 'r')
		ax2.tick_params('y', colors = 'r')

		fig.tight_layout()
		plt.show()
	elif method == 4:
		plt.plot(address.x, address.eset, 'ro', mfc='none', label='Avr. curve shift')
		plt.plot(address.x, address.peak, 'bo', mfc='none', label='WL peak shift')
		plt.plot(address.x, address.drv, 'go', mfc='none', label='Max. 1st deriv. shift')
		plt.legend(loc='upper right',ncol=1)
		plt.ylabel('Energy shift from GS to ES (eV)')
		plt.xlabel('Excited state fraction (%)')
		plt.show()
	elif method == 5:
		t = 0
		while t < 1:
			f = int(input("How much fraction is excited? "))
			address.esmodel = address.ravg[3]*(100/f) + address.ravg[2]
			plt.plot(address.ravg[0], address.ravg[2], 'b-', address.ravg[0], address.esmodel, 'r-')
			plt.ylabel('Fluorescence Yield (a.u.)')
			plt.xlabel('Energy (keV)')
			plt.show()
			plt.plot(address.ravg[0][:202], address.wdrv[f][0], 'b-', address.ravg[0][:202], address.wdrv[f][1], 'r-')
			plt.ylabel('First derivative (a.u.)')
			plt.xlabel('Energy (keV)')
			plt.show()
			
			ans = input("Are you satisfied with the esmodel? (y/n)")
			if ans == 'y':
				t += 1
				filedic[sample].append('esmodel')
			else: continue
	elif method == 6:
		fig, ax1 = plt.subplots()
		ax1.plot(address.ravg[0], address.esmodel, 'g-', address.ravg[0], address.ravg[2], 'b-')
		ax1.set_xlabel('Energy (keV)')
		ax1.set_ylabel('Fluorescence Yield (a.u.)', color='b')
		ax1.tick_params('y', colors = 'b')

		ax2 = ax1.twinx()
		ax2.plot(address.ravg[0], address.ravg[3], 'r-')
		ax2.set_ylabel('Transient (a.u.)', color = 'r')
		ax2.tick_params('y', colors = 'r')

		fig.tight_layout()
		plt.show()
	elif method == 7:
		plt.plot(address.dscan[0], address.dscan[1], 'ro', mfc='none')
		plt.plot(address.xfit, address.decayf, 'r--')
		print(address.popt[2])
		plt.show()
		
		
def gui():
	sample = 0
	while sample < len(dataset):
		filedic[sample] = ['raw']
		filedic[sample][0] = getdata(sample)
		print("{} sample loaded.".format(str(sample)))
		filedic[sample][0].errorcheck(sample)
		print("{} sample corrected.".format(str(sample)))
		if sample < 4:
			filedic[sample][0].excited(sample)
			print("{} sample excited fraction tested.".format(str(sample)))
			filedic[sample][0].exafs(sample)
			print("{} sample spline fitted.".format(str(sample)))	
		
		sample += 1
	
	sample = 0
	while sample < len(delayset):
		delaydic[sample] = ['raw']
		delaydic[sample][0] = delay(sample)
		print("{} sample delay loaded.".format(str(sample)))
		
		sample += 1
	
	feffdset = np.empty(2, '442f4')
	text = np.loadtxt(feff)	
	text = list(zip(*text))
	
	feffdset[0, :] = text[0]
	feffdset[0] = (feffdset[0]/1000) -0.015
	feffdset[1, :] = text[1]
	
	print("Hi, welcome to XANES, EXAFS analysis tool.")
	print("\nCurrently we have those items:\n0. Ru(phen)3\n1. Ru(phen)2dppz\n2. Ru(TAP)3\n3. Ru(TAP)2dppz\n4. Ru(phen)3 in water\n5. Ru(phen)2dppz in water\n6. Ru(TAP)2dppz in water")
		
	a = 0
	while a < 1:
		opt = int(input("\nPlease select an option\n0. Current filelist\n1. Plotting data\n2. feff on Ru(phen)3\n3. Saving data\n4. Exit\n"))
		if opt == 0:
			print("We have currently this sample list.")
			print(filedic)
		elif opt == 1:
			plot = [int(x) for x in input("\nPlease input sample # and plot method\n0. Ru foil in ion chamber\n1. All scans together\n2. Difference on all scans\n3. Laser ON/OFF and T\n4. Excited fraction scan\n5. Excited fraction choice with 1st Deriv.\n6. GS/ES and T\n7. Delay scan\n").split()]
			if plot[1] < 7:	plotting(filedic[plot[0]][0], plot[1], plot[0])
			else: plotting(delaydic[plot[0]][0], plot[1], plot[0])
		elif opt == 2:
			plt.plot(filedic[0][0].ravg[0], filedic[0][0].ravg[2], 'b-', feffdset[0], feffdset[1], 'r-')
			plt.title('FEFF vs. GS')
			plt.ylabel('Fluorescence Yield (a.u.)')
			plt.xlabel('Energy (keV)')
			plt.show()
		elif opt == 3:
			with open(saved, "w") as output:
				writer = csv.writer(output, lineterminator= '\n')
				writer.writerows(map(lambda x: [x], filedic[3][0].esmodel))
		else: a = 1

gui()

'''
plt.savefig(os.path.join(folder, 'myplot.png'))
'''