import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from openpyxl import load_workbook

# Datalist
RuMomonty = 'C:/data/bern/RuMo monty'
Rumonty = 'C:/data/bern/Ru monty'
Momonty = 'C:/data/bern/Mo monty'

RuMomaryam = 'C:/data/bern/RuMo maryam'
Rumaryam = 'C:/data/bern/Ru maryam'
RuMoMVmaryam = 'C:/data/bern/RuMoMV maryam'

s1comaryam = 'C:/data/bern/s1co maryam'
s1maryam = 'C:/data/bern/s1 maryam'
s3comaryam = 'C:/data/bern/s3co maryam'
s3maryam = 'C:/data/bern/s3 maryam'
s0comaryam = 'C:/data/bern/s0co maryam'
s0maryam = 'C:/data/bern/s0 maryam'
comaryam = 'C:/data/bern/co maryam'

news1co = 'C:/data/bern/news1co'
news1codiode = 'C:/data/bern/news1codiode'

s1copar = 'C:/data/bern/s1co parallel'
s1coper = 'C:/data/bern/s1co perpen'
s1par = 'C:/data/bern/s1 parallel'
s1per = 'C:/data/bern/s1 perpen'
s3copar = 'C:/data/bern/s3co parallel'
s3coper = 'C:/data/bern/s3co perpen'
s3par = 'C:/data/bern/s3 parallel'
s3per = 'C:/data/bern/s3 perpen'
s0copar = 'C:/data/bern/s0co parallel'
s0coper = 'C:/data/bern/s0co perpen'
s0par = 'C:/data/bern/s0 parallel'
s0per = 'C:/data/bern/s0 perpen'

cross1 = 'C:/data/bern/cross'

dyadbf = 'C:/data/bern/uvvis/dyadbf.csv'
dyadaf = 'C:/data/bern/uvvis/dyadaf.csv'
rubf = 'C:/data/bern/uvvis/rubf.csv'
ruaf = 'C:/data/bern/uvvis/ruaf.csv'
rumvbf = 'C:/data/bern/uvvis/rumvbf.csv'
rumvaf = 'C:/data/bern/uvvis/rumvaf.csv'

s1cobf = 'C:/data/bern/uvvis/s1cobf.csv'
s1coaf = 'C:/data/bern/uvvis/s1coaf.csv'
s1bf = 'C:/data/bern/uvvis/s1bf.csv'
s1af = 'C:/data/bern/uvvis/s1af.csv'
s3cobf = 'C:/data/bern/uvvis/s3cobf.csv'
s3coaf = 'C:/data/bern/uvvis/s3coaf.csv'
s3bf = 'C:/data/bern/uvvis/s3bf.csv'
s3af = 'C:/data/bern/uvvis/s3af.csv'
s0cobf = 'C:/data/bern/uvvis/s0cobf.csv'
s0coaf = 'C:/data/bern/uvvis/s0coaf.csv'
s0bf = 'C:/data/bern/uvvis/s0bf.csv'
s0af = 'C:/data/bern/uvvis/s0af.csv'
cobf = 'C:/data/bern/uvvis/cobf.csv'
coaf = 'C:/data/bern/uvvis/coaf.csv'

RuModiode = 'C:/data/bern/RuModiode'
Rudiode = 'C:/data/bern/Rudiode'
RuMoMVdiode = 'C:/data/bern/RuMoMVdiode'
s1codiode = 'C:/data/bern/s1codiode'
s1diode = 'C:/data/bern/s1diode'
s3codiode = 'C:/data/bern/s3codiode'
s3diode = 'C:/data/bern/s3diode'
s0codiode = 'C:/data/bern/s0codiode'
s0diode = 'C:/data/bern/s0diode'
codiode = 'C:/data/bern/codiode'
s1copardiode = 'C:/data/bern/s1copardiode'
s1coperdiode = 'C:/data/bern/s1coperdiode'
s1pardiode = 'C:/data/bern/s1pardiode'
s1perdiode = 'C:/data/bern/s1perdiode'
s3copardiode = 'C:/data/bern/s3copardiode'
s3coperdiode = 'C:/data/bern/s3coperdiode'
s3pardiode = 'C:/data/bern/s3pardiode'
s3perdiode = 'C:/data/bern/s3perdiode'
s0copardiode = 'C:/data/bern/s0copardiode'
s0coperdiode = 'C:/data/bern/s0coperdiode'
s0pardiode = 'C:/data/bern/s0pardiode'
s0perdiode = 'C:/data/bern/s0perdiode'

dataset = [RuMomaryam, Rumaryam, RuMoMVmaryam, s1comaryam, s1maryam, s3comaryam, s3maryam, s0comaryam, s0maryam, comaryam, s1copar, s1coper, s1par, s1per, s3copar, s3coper, s3par, s3per, s0copar, s0coper, s0par, s0per, news1co]
mdataset = [Momonty]
newset = [cross1]
diodeset = [RuModiode, Rudiode, RuMoMVdiode, s1codiode, s1diode, s3codiode, s3diode, s0codiode, s0diode, codiode, s1copardiode, s1coperdiode, s1pardiode, s1perdiode, s3copardiode, s3coperdiode, s3pardiode, s3perdiode, s0copardiode, s0coperdiode, s0pardiode, s0perdiode, news1codiode]

gvdpoints = [[[100,150,200,300,400],[16,24,32,46,60]], [[100,150,200,300,400],[22,30,38,52,64]], [[100,150,200,300,400],[18,26,34,49,62]], [[150,200,300,400,450],[25,33,49,62,67]], [[100,150,200,300,400],[21,29,38,53,65]], [[150,200,300,400,450],[21,29,44,58,64]], [[150,200,300,400,450],[28,36,51,63,69]], [[150,200,300,400,450],[21,29,44,57,62]], [[150,200,300,400,450],[20,28,43,57,62]], [[100,150,200,300,400],[14,23,31,47,61]], [[100,150,200,300,400],[31,35,40,47,54]], [[150,200,300,400,450],[38,42,49,55,57]], [[100,150,200,300,400],[33,38,42,49,55]], [[100,150,200,300,400],[33,37,41,49,55]], [[150,200,300,400,450],[30,34,43,51,54]], [[150,200,300,400,450],[29,34,42,49,52]], [[100,150,200,300,400],[28,32,36,44,51]], [[100,150,200,300,400],[28,33,37,45,52]], [[150,200,300,400,450],[15,19,27,34,37]], [[150,200,300,400,450],[16,20,28,35,37]], [[150,200,300,400,450],[14,18,26,33,36]], [[150,200,300,400,450],[16,20,28,35,37]], [[250,300,350,400,500],[22,31,38,45,58]], [[100,150,200,300,400],[34,42,50,64,77]], [[100,150,200,300,400],[25,34,42,58,72]]]
mgvdpoints = [[200,400,600,800,900],[27,36,42,45,46]]
vertical = [[75,450], [90,440], [75,450], [148,449], [187,425], [150,437], [150,447], [135,445], [160,447], [150,407], [140,438], [140,435], [184,414], [183,405], [134,453], [140,450], [105,447], [100,445], [120,445], [125,445], [125,435], [125,435], [255,512]]
pump = [[220,260], [220,250], [210,260], [215,255], [210,260], [220,270], [220,255], [213,260], [220,260], [205,260], [224,255], [220,250], [217,255], [215,260], [221,247], [224,246], [223,253], [221,249], [220,255], [220,255], [220,255], [220,255], [354,372]]
uvvisset = [[dyadbf,dyadaf], [rubf,ruaf], [rumvbf,rumvaf], [s1cobf,s1coaf], [s1bf,s1af], [s3cobf,s3coaf], [s3bf,s3af], [s0cobf,s0coaf], [s0bf,s0af], [cobf,coaf], [s1cobf,s1coaf], [s1cobf,s1coaf], [s1bf,s1af], [s1bf,s1af], [s3cobf,s3coaf], [s3cobf,s3coaf], [s3bf,s3af], [s3bf,s3af], [s0cobf,s0coaf], [s0cobf,s0coaf], [s0bf,s0af], [s0bf,s0af], [s1cobf,s1coaf]]
conc = [[1],[1],[1],[1],[1],[0.009*0.1],[0.002541*0.1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]]

wl = np.round(np.loadtxt('C:/data/bern/wl_cal.txt'))
wl_aniso = np.round(np.loadtxt('C:/data/bern/wl_cal_aniso.txt'))
mdelay = np.loadtxt('C:/data/bern/monty/delays_sh.txt')
mwl = np.loadtxt('C:/data/bern/monty/wl_cal.txt')[::-1]
filedic = {}
mfiledic = {}

#class browser:
#	pass

def erfunc(x, t):
	return np.exp((0.07/1.66511/2/t)**2 - x/t)*(erf(x/0.07*1.66511-(0.07/1.66511/2/t))+1)/2

def igorf(x, a1, a2, a3, t1, t2, t3):
	return a1*erfunc(x,t1)+a2*erfunc(x,t2)+a3*erfunc(x,t3)#+a4*erfunc(x,t4) # number of timeconstants

def subnoise(set, sample):
    temp=np.empty([len(set),len(set[0])])
    for a,wave in enumerate(set): #fixed wavelength, time-dependent
        waveavr=np.average(wave)
        for b,num in enumerate(wave):
            if waveavr-0.025<num<waveavr+0.025:
                temp[a][b]=num
            elif b==0:
                temp[a][b]=wave[b+1]
            elif b==len(wave)-1:
                temp[a][b]=wave[b-1]
            else:
                temp[a][b]=(wave[b-1]+wave[b+1])/2

    temp2=np.empty([len(set[0]), len(set)]) #fixed time, wavelength changes
    for c,wave in enumerate(np.transpose(temp)):
        for d,num in enumerate(wave):
            if d<3:
                waveavr=np.average(wave[:d+3])
            elif d>len(wave)-3:
                waveavr=np.average(wave[d-3:])
            else:
                waveavr=np.average(wave[d-3:d+3])

            if waveavr-0.0005<num<waveavr+0.0005:
                temp2[c][d]=num
            elif pump[sample][0]-vertical[sample][0]-3<d<pump[sample][0]-vertical[sample][0]+3:
                temp2[c][d]=num
            else:
                temp2[c][d]=waveavr

    return np.transpose(temp2)

class maryam:
    def __init__(self, sample, address):
        files = os.listdir(address)
        self.dset = [None]*(len(files))
        self.delay = [None]*(len(files))
        #self.nbg = [None]*(len(files))
        filedic[sample].append('dset')
        filedic[sample].append('delay')

        for num, file in enumerate(files):
            #diode = np.loadtxt(os.path.join(diodeset[sample], file), skiprows=1).transpose()[1] #Ch1_unpumped as I0, or it's already normalized
            sheet = np.loadtxt(os.path.join(address, file), skiprows = 2)
            self.delay[num] = sheet.transpose()[0]
            self.dset[num] = sheet.transpose()[2:][::-1] # raw file
            #self.nbg[num] = (sheet - np.average(sheet[:5], axis=0)).transpose()[2:][::-1]#/diode # background subtracted, do we need this?

        if len(files)>1:
            assert np.allclose(self.delay[0], self.delay[-1])

        #filedic[sample].append('nbg')

    def uvvis(self, sample):
        self.bfuv=(np.loadtxt(uvvisset[sample][0], delimiter=';', skiprows=1)).transpose()
        self.afuv=(np.loadtxt(uvvisset[sample][1], delimiter=';', skiprows=1)).transpose()

        filedic[sample].append('uvvis')

    def odcali(self, sample):
        self.cali = np.average(np.array(self.dset), axis=0)/self.afuv[1][375] #Averaged and OD normalized; is this correct?
        #plt.imshow(self.cali, cmap='terrain', aspect="auto", clim=(-0.01, 0.02), origin='lower')
        #plt.show()

        filedic[sample].append('cali')

    def nogvd(self, sample):
        coef=np.polyfit(gvdpoints[sample][0], gvdpoints[sample][1], 3)
        self.timezero=list(self.delay[0]).index(0)

        self.ngvd=[None]*len(self.cali)
        for i,wave in enumerate(self.cali):
            rex=int(np.round((coef[0])*i*i*i+(coef[1])*i*i+(coef[2])*i+coef[3]))
            if len(wave)>100:
                fac=100
            elif len(wave)>90:
                fac=65
            else:
                fac=47

            if rex>self.timezero:
                self.ngvd[i]=np.concatenate((wave[(rex-self.timezero):fac], np.repeat(wave[fac],(rex-self.timezero)), wave[fac:]), axis=None)
            else:
                self.ngvd[i]=wave #GVD correction

        filedic[sample].append('ngvd')

        #CHECK IRF ?

        #print(self.timezero)
        #plt.plot(np.array(self.ngvd).transpose()[0][pump[sample][0]:pump[sample][1]],'b')
        #plt.show()
        #plt.plot([list(scan[pump[sample][0]:pump[sample][1]]).index(max(scan[pump[sample][0]:pump[sample][1]])) for scan in np.array(self.ngvd).transpose()], 'b')
        #plt.show()
        #plt.plot([np.trapz(scan[pump[sample][0]:pump[sample][1]]) for scan in np.array(self.ngvd).transpose()], 'b')
        #plt.show()

    def svdecomposition(self, sample):
        vcutting = np.concatenate((self.ngvd[vertical[sample][0]:pump[sample][0]],self.ngvd[pump[sample][1]:vertical[sample][1]]), axis=0)
        if len(self.ngvd[0]) > 110:
            self.wl = np.append(wl[vertical[sample][0]:pump[sample][0]],wl[pump[sample][1]:vertical[sample][1]])
        else:
            self.wl = np.append(wl_aniso[vertical[sample][0]:pump[sample][0]],wl_aniso[pump[sample][1]:vertical[sample][1]])

        hcutting = np.concatenate((vcutting.transpose()[:self.timezero-3],vcutting.transpose()[self.timezero+3:]), axis=0)
        self.sdelay = np.concatenate((self.delay[0][:self.timezero-3], self.delay[0][self.timezero+3:]), axis=0)
        self.svd = np.transpose(hcutting) # cut, still with noise

		#self.sdelay = self.delay[0]
        #self.svd = subnoise(self.cut, sample) DO WE NEED THIS?

        self.u, s, self.vh = np.linalg.svd(self.svd, full_matrices=False)
        assert np.allclose(self.svd, np.dot(self.u, np.dot(np.diag(s), self.vh)))
        self.s = tuple(s)
        s[2:] = 0 # number of singular values
        filedic[sample].append('svd')

        self.ss = s
        self.rec = np.dot(self.u, (np.dot(np.diag(s), self.vh)))
        self.gfit = self.vh[:2] # number of singular values
        self.bdas = (self.u*s).transpose()[:2] # number of singular values
        self.diff = self.svd - self.rec

    def globalfit(self, sample, t):
        self.decay = [None]*2 # number of singular values
        self.adas = [None]*3 # number of timeconstants
        temp = [None]*2 # number of singular values

        def tempf(x, a1, a2, a3):
            return igorf(x, a1, a2, a3, t[0], t[1], t[2]) # number of timeconstants
        fopt, fcov = curve_fit(tempf, self.sdelay, self.gfit[0], p0=(0, 0, 0)) # number of timeconstants

        def tempf2(x, t1, t2):
            return igorf(x, fopt[0], fopt[1], fopt[2], t1, t2, t[2]) # number of timeconstants
        sopt, scov = curve_fit(tempf2, self.sdelay, self.gfit[0], p0=(t[0], t[1])) # number of timeconstants
        print(sopt)
        print(np.sqrt(np.diag(scov)))

        self.t = [sopt, np.sqrt(np.diag(scov))]
        def tempf3(x, a11, a21, a31):
            return igorf(x, a11, a21, a31, sopt[0], sopt[1], t[2]) # number of timeconstants

        for i, scan in enumerate(self.gfit):
            popt, pcov = curve_fit(tempf3, self.sdelay, scan, p0=(0, 0, 0)) # number of timeconstants
            self.decay[i] = tempf3(self.sdelay, *popt)
            residuals = scan - tempf3(self.sdelay, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((scan - np.mean(scan))**2)
            r_squared = 1 - (ss_res/ss_tot) # double-check needed

            print(popt)
            print(r_squared)
            temp[i] = popt

        temp1 = np.array(self.decay + self.vh[2:].tolist()) # number of singular values
        self.final =  np.dot(self.u, (np.dot(np.diag(self.ss), temp1)))
        self.finalfit = self.svd - self.final

        plt.imshow(self.finalfit, cmap='terrain', aspect="auto", clim=(-0.005, 0.01), origin='lower')
        plt.colorbar()
        plt.show()
        for j, time in enumerate(t):
            self.adas[j] = temp[0][j]*self.bdas[0] + temp[1][j]*self.bdas[1] # number of singular values

        filedic[sample].append('gfit')
        filedic[sample].append('das')

class monty:
	def __init__(self, sample, address):
		files = os.listdir(address)
		self.dset = [None]*(len(files))

		for l, file in enumerate(files):
			fullpath = os.path.join(address, file)
			wb = load_workbook(fullpath)
			sheet = wb['Sheet1']
			allcell = sheet['A1':'EY1024']
			dset = [None]*1024

			for m, row in enumerate(allcell):
				wave = [None]*155
				for k, cell in enumerate(row):
					wave[k] = cell.value
				dset[m] = wave

			for i, wave in enumerate(dset):
				for j, num in enumerate(dset[i]):
					if num == None:	dset[i][j] = 0
					elif num > 10: dset[i][j] = num/1000
					else: continue

			self.dset[l] = dset

	def nobg(self, sample):
		self.nbg = bgsubtract(self.dset, 0, 4)
		mfiledic[sample].append('nobg')

	def nogvd(self, sample):
		coef = np.polyfit(mgvdpoints[0], mgvdpoints[1], 3)
		#self.timezero = 9
		#for k, time in enumerate(mdelay):
		#	if time == 0: self.timezero = k
		#	else: continue

		self.ngvd = [None]*(len(self.nbg))

		for j, scan in enumerate(self.nbg):
			ngvd = [None]*1024
			for i, wave in enumerate(scan):
				x = (coef[0])*i*i*i + (coef[1])*i*i + (coef[2])*i + coef[3]
				rex = ceiling(x)
				if rex > 19:
					ngvd[i] = wave[(rex-19):79] + (rex-19)*[wave[79]] + wave[79:]
				else: ngvd[i] = wave

			self.ngvd[j] = ngvd

		mfiledic[sample].append('ngvd')

		npset = np.array(self.ngvd)
		self.cali = np.sum(npset, axis=0)/len(npset)
		'''
		plt.imshow(self.cali, cmap='terrain', aspect="auto", clim=(-0.025, 0.02), origin='lower')
		idy = np.linspace(0,len(mwl)-1,11).astype('int')
		plt.yticks(idy, [mwl[i] for i in idy])
		idx = np.linspace(0,len(mdelay)-1,11).astype('int')
		plt.xticks(idx, [mdelay[i] for i in idx])
		plt.colorbar()
		plt.show()
		'''

def plotting(address, method):
	if method == 0:
		for scan in address.nbg:
			tscan = list(map(list, zip(*scan)))
			plt.plot(wl[135:450], tscan[3][135:450], 'r-')
			plt.show()
	elif method == 1:
		for scan in address.ngvd:
			tscan = list(map(list, zip(*scan)))
			plt.plot(wl[135:450], tscan[3][135:450], 'r-')
			plt.show()
	elif method == 2:
		plt.plot(address.s, 'ro', mfc='none')
		plt.ylabel('Singular Value (a.u.)')
		plt.xlabel('Index')
		plt.show()
	elif method == 3:
		for i, timeconstant in enumerate(address.gfit):
			plt.plot(address.sdelay, timeconstant, 'r-', address.sdelay, address.decay[i], 'k-')
			plt.title('Global fitting_kinetics')
			plt.ylabel('Intensity (a.u.)')
			plt.xlabel('Time delay (ps)')
			plt.show()
	elif method == 4:
		for i, scan in enumerate(address.gfit):
			plt.plot(address.wl, address.bdas[i], 'r-')
			plt.title('Global fitting_spectra')
			plt.ylabel('Absorbance (OD)')
			plt.xlabel('Wavelength (nm)')
			plt.show()

		#np.savetxt('test.txt', (address.wl, address.bdas[0]))
	elif method == 5:
		plt.plot(address.bfuv[0], address.bfuv[1], 'r-', address.afuv[0], address.afuv[1], 'b-')
		plt.title('UV/vis spectra (red: before TAS, blue: after TAS)')
		plt.ylabel('Absorbance (OD)')
		plt.xlabel('Wavelength (nm)')
		plt.show()
	elif method == 6:
		temp = np.transpose(address.cut)
		if len(temp) > 100:
			plt.plot(address.wl, temp[5], 'r-', label=(str(address.sdelay[5]) + ' ps'))
			plt.plot(address.wl, temp[11], 'y-', label=(str(address.sdelay[11]) + ' ps'))
			plt.plot(address.wl, temp[29], 'g-', label=(str(address.sdelay[29]) + ' ps'))
			plt.plot(address.wl, temp[123], 'c-', label=(str(address.sdelay[123]) + ' ps'))
			plt.plot(address.wl, temp[143], 'b-', label=(str(address.sdelay[143]) + ' ps'))
			plt.plot(address.wl, temp[155], 'm-', label=(str(address.sdelay[155]) + ' ps'))
			plt.legend(loc='upper right',ncol=2)
			plt.title('Spectra at different time delay')
			plt.ylabel('Absorbance (OD)')
			plt.xlabel('Wavelength (nm)')
			plt.show()
		#plt.plot(address.wl, temp[11] - temp[155], 'y-', label=(str(address.sdelay[11]) + ' ps - ' + str(address.sdelay[155]) + ' ps'))
		#plt.plot(address.wl, temp[29] - temp[155], 'g-', label=(str(address.sdelay[29]) + ' ps - ' + str(address.sdelay[155]) + ' ps'))
		#plt.plot(address.wl, temp[123] - temp[155], 'c-', label=(str(address.sdelay[123]) + ' ps - ' + str(address.sdelay[155]) + ' ps'))
		#plt.plot(address.wl, temp[143] - temp[155], 'b-', label=(str(address.sdelay[143]) + ' ps - ' + str(address.sdelay[155]) + ' ps'))
		#plt.legend(loc='upper right',ncol=2)
		#plt.title('Spectra at different time delay')
		#plt.ylabel('Absorbance (OD)')
		#plt.xlabel('Wavelength (nm)')
		#plt.show()
		elif len(temp) > 80:
			plt.plot(address.wl, temp[11], 'r-', label=(str(address.sdelay[11]) + ' ps'))
			plt.plot(address.wl, temp[23], 'y-', label=(str(address.sdelay[23]) + ' ps'))
			plt.plot(address.wl, temp[34], 'g-', label=(str(address.sdelay[34]) + ' ps'))
			plt.plot(address.wl, temp[63], 'c-', label=(str(address.sdelay[63]) + ' ps'))
			plt.plot(address.wl, temp[72], 'b-', label=(str(address.sdelay[72]) + ' ps'))
			plt.plot(address.wl, temp[87], 'm-', label=(str(address.sdelay[87]) + ' ps'))
			plt.legend(loc='upper right',ncol=2)
			plt.title('Spectra at different time delay')
			plt.ylabel('Absorbance (OD)')
			plt.xlabel('Wavelength (nm)')
			plt.show()
		else:
			plt.plot(address.wl, temp[7], 'r-', label=(str(address.sdelay[7]) + ' ps'))
			plt.plot(address.wl, temp[9], 'y-', label=(str(address.sdelay[9]) + ' ps'))
			plt.plot(address.wl, temp[17], 'g-', label=(str(address.sdelay[17]) + ' ps'))
			plt.plot(address.wl, temp[48], 'c-', label=(str(address.sdelay[48]) + ' ps'))
			plt.plot(address.wl, temp[57], 'b-', label=(str(address.sdelay[57]) + ' ps'))
			plt.plot(address.wl, temp[72], 'm-', label=(str(address.sdelay[72]) + ' ps'))
			plt.legend(loc='upper right',ncol=2)
			plt.title('Spectra at different time delay')
			plt.ylabel('Absorbance (OD)')
			plt.xlabel('Wavelength (nm)')
			plt.show()
	elif method == 7:
		temp = np.transpose(address.svd)
		if len(temp) > 100:
			plt.plot(address.wl, temp[5], 'r-', label=(str(address.sdelay[5]) + ' ps'))
			plt.plot(address.wl, temp[11], 'y-', label=(str(address.sdelay[11]) + ' ps'))
			plt.plot(address.wl, temp[29], 'g-', label=(str(address.sdelay[29]) + ' ps'))
			plt.plot(address.wl, temp[123], 'c-', label=(str(address.sdelay[123]) + ' ps'))
			plt.plot(address.wl, temp[143], 'b-', label=(str(address.sdelay[143]) + ' ps'))
			plt.plot(address.wl, temp[155], 'm-', label=(str(address.sdelay[155]) + ' ps'))
			plt.legend(loc='upper right',ncol=2)
			plt.title('Spectra at different time delay')
			plt.ylabel('Absorbance (OD)')
			plt.xlabel('Wavelength (nm)')
			plt.show()

			plt.plot(address.wl, temp[11] - temp[155], 'y-', label=(str(address.sdelay[11]) + ' ps - ' + str(address.sdelay[155]) + ' ps'))
			plt.plot(address.wl, temp[29] - temp[155], 'g-', label=(str(address.sdelay[29]) + ' ps - ' + str(address.sdelay[155]) + ' ps'))
			plt.plot(address.wl, temp[123] - temp[155], 'c-', label=(str(address.sdelay[123]) + ' ps - ' + str(address.sdelay[155]) + ' ps'))
			plt.plot(address.wl, temp[143] - temp[155], 'b-', label=(str(address.sdelay[143]) + ' ps - ' + str(address.sdelay[155]) + ' ps'))
			plt.legend(loc='upper right',ncol=2)
			plt.title('Spectra at different time delay')
			plt.ylabel('Absorbance (OD)')
			plt.xlabel('Wavelength (nm)')
			plt.show()
		elif len(temp) > 80:
			plt.plot(address.wl, temp[11], 'r-', label=(str(address.sdelay[11]) + ' ps'))
			plt.plot(address.wl, temp[23], 'y-', label=(str(address.sdelay[23]) + ' ps'))
			plt.plot(address.wl, temp[34], 'g-', label=(str(address.sdelay[34]) + ' ps'))
			plt.plot(address.wl, temp[63], 'c-', label=(str(address.sdelay[63]) + ' ps'))
			plt.plot(address.wl, temp[72], 'b-', label=(str(address.sdelay[72]) + ' ps'))
			plt.plot(address.wl, temp[87], 'm-', label=(str(address.sdelay[87]) + ' ps'))
			plt.legend(loc='upper right',ncol=2)
			plt.title('Spectra at different time delay')
			plt.ylabel('Absorbance (OD)')
			plt.xlabel('Wavelength (nm)')
			plt.show()
		else:
			plt.plot(address.wl, temp[7], 'r-', label=(str(address.sdelay[7]) + ' ps'))
			plt.plot(address.wl, temp[9], 'y-', label=(str(address.sdelay[9]) + ' ps'))
			plt.plot(address.wl, temp[17], 'g-', label=(str(address.sdelay[17]) + ' ps'))
			plt.plot(address.wl, temp[48], 'c-', label=(str(address.sdelay[48]) + ' ps'))
			plt.plot(address.wl, temp[57], 'b-', label=(str(address.sdelay[57]) + ' ps'))
			plt.plot(address.wl, temp[72], 'm-', label=(str(address.sdelay[72]) + ' ps'))
			plt.legend(loc='upper right',ncol=2)
			plt.title('Spectra at different time delay')
			plt.ylabel('Absorbance (OD)')
			plt.xlabel('Wavelength (nm)')
			plt.show()
	elif method == 8:
		temp = np.transpose(address.rec)
		if len(temp) > 100:
			plt.plot(address.wl, temp[2], 'r-', label=(str(address.sdelay[2]) + ' ps'))
			plt.plot(address.wl, temp[11], 'y-', label=(str(address.sdelay[11]) + ' ps'))
			plt.plot(address.wl, temp[29], 'g-', label=(str(address.sdelay[29]) + ' ps'))
			plt.plot(address.wl, temp[123], 'c-', label=(str(address.sdelay[123]) + ' ps'))
			plt.plot(address.wl, temp[143], 'b-', label=(str(address.sdelay[143]) + ' ps'))
			plt.plot(address.wl, temp[152], 'm-', label=(str(address.sdelay[152]) + ' ps'))
			plt.legend(loc='upper right',ncol=2)
			plt.title('Spectra at different time delay')
			plt.ylabel('Absorbance (OD)')
			plt.xlabel('Wavelength (nm)')
			plt.show()
		elif len(temp) > 80:
			plt.plot(address.wl, temp[11], 'r-', label=(str(address.sdelay[11]) + ' ps'))
			plt.plot(address.wl, temp[23], 'y-', label=(str(address.sdelay[23]) + ' ps'))
			plt.plot(address.wl, temp[34], 'g-', label=(str(address.sdelay[34]) + ' ps'))
			plt.plot(address.wl, temp[63], 'c-', label=(str(address.sdelay[63]) + ' ps'))
			plt.plot(address.wl, temp[72], 'b-', label=(str(address.sdelay[72]) + ' ps'))
			plt.plot(address.wl, temp[87], 'm-', label=(str(address.sdelay[87]) + ' ps'))
			plt.legend(loc='upper right',ncol=2)
			plt.title('Spectra at different time delay')
			plt.ylabel('Absorbance (OD)')
			plt.xlabel('Wavelength (nm)')
			plt.show()
		else:
			plt.plot(address.wl, temp[7], 'r-', label=(str(address.sdelay[7]) + ' ps'))
			plt.plot(address.wl, temp[9], 'y-', label=(str(address.sdelay[9]) + ' ps'))
			plt.plot(address.wl, temp[17], 'g-', label=(str(address.sdelay[17]) + ' ps'))
			plt.plot(address.wl, temp[48], 'c-', label=(str(address.sdelay[48]) + ' ps'))
			plt.plot(address.wl, temp[57], 'b-', label=(str(address.sdelay[57]) + ' ps'))
			plt.plot(address.wl, temp[72], 'm-', label=(str(address.sdelay[72]) + ' ps'))
			plt.legend(loc='upper right',ncol=2)
			plt.title('Spectra at different time delay')
			plt.ylabel('Absorbance (OD)')
			plt.xlabel('Wavelength (nm)')
			plt.show()

		#np.savetxt('C:/Users/choit/Desktop/Veusz/IrLCo/spectra.txt', (address.wl, temp[2], temp[11], temp[29], temp[123], temp[143], temp[152]))
	elif method == 9:
		plt.imshow(address.rec, cmap='terrain', aspect="auto", clim=(-0.035, 0.03), origin='lower')
		plt.colorbar()
		plt.show()
		kinetic = [int(x) for x in input("Give 4 index numbers for kinetic traces ").split()]
		plt.plot(address.sdelay, address.rec[kinetic[0]], 'ro', mfc='none', label=(str(address.wl[kinetic[0]]) + ' nm'))
		#plt.plot(address.sdelay, address.cut[kinetic[1]], 'yo', mfc='none', label=(str(address.wl[kinetic[1]]) + ' nm'))
		plt.plot(address.sdelay, address.rec[kinetic[1]], 'go', mfc='none', label=(str(address.wl[kinetic[1]]) + ' nm'))
		plt.plot(address.sdelay, address.rec[kinetic[3]], 'co', mfc='none', label=(str(address.wl[kinetic[3]]) + ' nm'))
		plt.plot(address.sdelay, address.rec[kinetic[2]], 'bo', mfc='none', label=(str(address.wl[kinetic[2]]) + ' nm'))
		#plt.plot(address.sdelay, address.cut[kinetic[5]], 'mo', mfc='none', label=(str(address.wl[kinetic[5]]) + ' nm'))

		plt.plot(address.sdelay, address.final[kinetic[0]], 'r--')
		#plt.plot(address.sdelay, address.final[kinetic[1]], 'y--')
		plt.plot(address.sdelay, address.final[kinetic[1]], 'g--')
		plt.plot(address.sdelay, address.final[kinetic[3]], 'c--')
		plt.plot(address.sdelay, address.final[kinetic[2]], 'b--')
		#plt.plot(address.sdelay, address.final[kinetic[5]], 'm--')
		plt.legend(loc='upper right',ncol=3)
		plt.title('Kinetic traces at different probe wavelength')
		plt.ylabel('TA (a.u.)')
		plt.xlabel('Time delay (ps)')
		plt.yscale('symlog')
		plt.ylim(-0.02, 0.035)
		plt.show()

		#np.savetxt('C:/Users/choit/Desktop/Veusz/IrLCo/kinetics.txt', (address.sdelay, address.rec[kinetic[0]], address.rec[kinetic[1]], address.rec[kinetic[2]], address.rec[kinetic[3]], address.final[kinetic[0]], address.final[kinetic[1]], address.final[kinetic[2]], address.final[kinetic[3]]))
	elif method == 10:
		plt.imshow(address.svd, cmap='terrain', aspect="auto", clim=(-0.035, 0.03), origin='lower')
		plt.colorbar()
		plt.show()
		kinetic = [int(x) for x in input("Give 4 index numbers for kinetic traces ").split()]
		plt.plot(address.sdelay, address.svd[kinetic[0]], 'ro', mfc='none', label=(str(address.wl[kinetic[0]]) + ' nm'))
		#plt.plot(address.sdelay, address.svd[kinetic[1]], 'yo', mfc='none', label=(str(address.wl[kinetic[1]]) + ' nm'))
		plt.plot(address.sdelay, address.svd[kinetic[1]], 'go', mfc='none', label=(str(address.wl[kinetic[1]]) + ' nm'))
		plt.plot(address.sdelay, address.svd[kinetic[3]], 'co', mfc='none', label=(str(address.wl[kinetic[3]]) + ' nm'))
		plt.plot(address.sdelay, address.svd[kinetic[2]], 'bo', mfc='none', label=(str(address.wl[kinetic[2]]) + ' nm'))
		#plt.plot(address.sdelay, address.svd[kinetic[5]], 'mo', mfc='none', label=(str(address.wl[kinetic[5]]) + ' nm'))

		plt.plot(address.sdelay, address.rec[kinetic[0]], 'r--')
		#plt.plot(address.sdelay, address.rec[kinetic[1]], 'y--')
		plt.plot(address.sdelay, address.rec[kinetic[1]], 'g--')
		plt.plot(address.sdelay, address.rec[kinetic[3]], 'c--')
		plt.plot(address.sdelay, address.rec[kinetic[2]], 'b--')
		#plt.plot(address.sdelay, address.rec[kinetic[5]], 'm--')
		plt.legend(loc='upper right',ncol=3)
		plt.title('Kinetic traces at different probe wavelength')
		plt.ylabel('TA (a.u.)')
		plt.xlabel('Time delay (ps)')
		#plt.xscale('symlog')
		plt.ylim(-0.02, 0.035)
		plt.show()

		#np.savetxt('C:/Users/choit/Desktop/Veusz/IrL/kinetics_par.txt', (address.sdelay, address.svd[kinetic[0]], address.svd[kinetic[1]], address.svd[kinetic[2]], address.svd[kinetic[3]], address.rec[kinetic[0]], address.rec[kinetic[1]], address.rec[kinetic[2]], address.rec[kinetic[3]]))
	elif method == 11:
		for i, scan in enumerate(address.gfit):
			fig = plt.figure()
			ax1 = fig.add_subplot(1, 2, 1)
			ax1.plot(address.wl, address.bdas[i], 'r-')
			ax1.set_ylabel('Absorbance (OD)')
			ax1.set_xlabel('Wavelength (nm)')

			ax2 = fig.add_subplot(1, 2, 2)
			ax2.plot(address.sdelay, scan, 'r-')
			ax2.yaxis.tick_right()
			ax2.set_xlabel('Time delay (ps)')
			plt.show()
	elif method == 12: # number of timeconstants
		plt.plot(address.wl, address.adas[0], 'r-', label=('{:.2f} +-'.format(address.t[0][0]) + ' {:.2f} ps'.format(address.t[1][0])))
		plt.plot(address.wl, address.adas[1], 'y-', label=('{:.2f} +-'.format(address.t[0][1]) + ' {:.2f} ps'.format(address.t[1][1])))
		#plt.plot(address.wl, address.adas[2], 'g-', label=('{:.2f} +-'.format(address.t[0][2]) + ' {:.2f} ps'.format(address.t[1][2])))
		#plt.plot(address.wl, address.adas[3], 'b-', label=('{:.2f} +-'.format(address.t[0][3]) + ' {:.2f} ps'.format(address.t[1][3])))
		plt.plot(address.wl, address.adas[2], 'b-', label='long living')
		plt.legend(loc='upper right',ncol=2)
		plt.title('Decay-associated spectra (DAS)')
		plt.ylabel('Absorbance (OD)')
		plt.xlabel('Wavelength (nm)')
		plt.show()
		#plt.plot(address.wl, address.adas[0] + address.adas[4] + address.adas[3] + address.adas[2] + address.adas[1], 'r-', label=('DAS_C5C4C3C2C1'))
		#plt.plot(address.wl, address.adas[3] + address.adas[0] + address.adas[2] + address.adas[1], 'g-', label=('DAS_C5C4C3C2'))
		plt.plot(address.wl, address.adas[2] + address.adas[1] + address.adas[0], 'r-', label=('DAS_C5C4C3'))
		plt.plot(address.wl, address.adas[2] + address.adas[1], 'y-', label=('DAS_C5C4'))
		plt.plot(address.wl, address.adas[2], 'b-', label=('DAS_C5'))
		plt.legend(loc='upper right',ncol=2)
		plt.title('Decay-associated spectra (DAS)')
		plt.ylabel('Absorbance (OD)')
		plt.xlabel('Wavelength (nm)')
		plt.show()

		#np.savetxt('C:/Users/choit/Desktop/Veusz/IrLCo/daspar.txt', (address.wl, address.adas[0], address.adas[1], address.adas[2]))
	elif method == 13:
		for scan in address.nbg:
			plt.imshow(scan, cmap='terrain', aspect="auto", clim=(-0.025, 0.02), origin='lower')
			idy = np.linspace(0,len(wl)-1,11).astype('int')
			plt.yticks(idy, [wl[i] for i in idy])
			idx = np.linspace(0,len(address.delay[0])-1,11).astype('int')
			plt.xticks(idx, [address.delay[0][i] for i in idx])
			plt.colorbar()
			plt.show()
	elif method == 14:
		for scan in address.ngvd:
			plt.imshow(scan, cmap='terrain', aspect="auto", clim=(-0.025, 0.02), origin='lower')
			idy = np.linspace(0,len(wl)-1,11).astype('int')
			plt.yticks(idy, [wl[i] for i in idy])
			idx = np.linspace(0,len(address.delay[0])-1,11).astype('int')
			plt.xticks(idx, [address.delay[0][i] for i in idx])
			plt.colorbar()
			plt.show()
	elif method == 15:
		plt.imshow(address.cali, cmap='terrain', aspect="auto", clim=(-0.035, 0.03), origin='lower')
		idy = np.linspace(0,len(wl)-1,11).astype('int')
		plt.yticks(idy, [wl[i] for i in idy])
		idx = np.linspace(0,len(address.delay[0])-1,11).astype('int')
		plt.xticks(idx, [address.delay[0][i] for i in idx])
		plt.colorbar()
		plt.title('GVD corrected, OD calibrated TAS without background')
		plt.ylabel('Wavelength (nm)')
		plt.xlabel('Time delay (ps)')
		plt.show()
	elif method == 16:
		plt.imshow(address.svd, cmap='terrain', aspect="auto", clim=(-0.035, 0.03), origin='lower')
		idy = np.linspace(0,len(address.wl)-1,11).astype('int')
		plt.yticks(idy, [address.wl[i] for i in idy])
		idx = np.linspace(0,len(address.sdelay)-1,11).astype('int')
		plt.xticks(idx, [address.sdelay[i] for i in idx])
		plt.colorbar()
		plt.title('Before SVD with noise subtraction')
		plt.ylabel('Wavelength (nm)')
		plt.xlabel('Time delay (ps)')
		plt.show()

		#np.savetxt('C:/Users/choit/Desktop/Veusz/IrthCo/IrthCo_delay.dat', (address.sdelay), fmt='%7.6f', delimiter=' ')
	elif method == 17:
		plt.imshow(address.rec, cmap='terrain', aspect="auto", clim=(-0.035, 0.03), origin='lower')
		idy = np.linspace(0,len(address.wl)-1,11).astype('int')
		plt.yticks(idy, [address.wl[i] for i in idy])
		idx = np.linspace(0,len(address.sdelay)-1,11).astype('int')
		plt.xticks(idx, [address.sdelay[i] for i in idx])
		plt.colorbar()
		plt.title('Reconstructed TAS after SVD')
		plt.ylabel('Wavelength (nm)')
		plt.xlabel('Time delay (ps)')
		plt.show()
	elif method == 18:
		plt.imshow(address.diff, cmap='terrain', aspect="auto", clim=(-0.005, 0.01), origin='lower')
		idy = np.linspace(0,len(address.wl)-1,11).astype('int')
		plt.yticks(idy, [address.wl[i] for i in idy])
		idx = np.linspace(0,len(address.sdelay)-1,11).astype('int')
		plt.xticks(idx, [address.sdelay[i] for i in idx])
		plt.colorbar()
		plt.title('Residual (data - reconstructed data from SVD)')
		plt.ylabel('Wavelength (nm)')
		plt.xlabel('Time delay (ps)')
		plt.show()
	elif method == 19:
		plt.imshow(address.finalfit, cmap='terrain', aspect="auto", clim=(-0.005, 0.01), origin='lower')
		idy = np.linspace(0,len(address.wl)-1,11).astype('int')
		plt.yticks(idy, [address.wl[i] for i in idy])
		idx = np.linspace(0,len(address.sdelay)-1,11).astype('int')
		plt.xticks(idx, [address.sdelay[i] for i in idx])
		plt.colorbar()
		plt.title('Residual (data - reconstructed data from DASs)')
		plt.ylabel('Wavelength (nm)')
		plt.xlabel('Time delay (ps)')
		plt.show()
	else: pass

def gui():
	print("Hi, welcome to TAS & UV/vis analysis tool.")
	print("\nCurrently we have those items:\n(Maryam)\n0. Ru-Mo dyad\n1. Ru- moiety\n2. Ru-Mo with MV\n3. Ir-O-Co\n4. Ir-O\n5. Ir-th-Co\n6. Ir-th\n7. Ir-L-Co\n8. Ir-L\n9. Co\n\n10. Ir-O-Co parallel\n11. Ir-O-Co perpendicular\n12. Ir-O parallel\n13. Ir-O perpendicular\n14. Ir-th-Co parallel\n15. Ir-th-Co perpendicular\n16. Ir-th parallel\n17. Ir-th perpendicular\n18. Ir-L-Co parallel\n19. Ir-L-Co perpendicular\n20. Ir-L parallel\n21. Ir-L perpendicular\n22. 2020 Ir-O-Co\nStart analyzing maryam dataset...\n")
	sample = 0
	while sample < len(dataset):
		filedic[sample] = ['raw']
		filedic[sample][0] = maryam(sample, dataset[sample])
		print("{} sample loaded.".format(str(sample)))
		filedic[sample][0].uvvis(sample)
		print("{} sample UV/vis loaded.".format(str(sample)))
		filedic[sample][0].odcali(sample)
		print("{} sample OD calibrated.".format(str(sample)))
		filedic[sample][0].nogvd(sample)
		print("{} sample GVD corrected.".format(str(sample)))
		filedic[sample][0].svdecomposition(sample)
		print("{} sample SVD completed.".format(str(sample)))
		sample += 1

	sample = 0
	#mfiledic[sample] = ['raw']
	#mfiledic[sample][0] = monty(sample, mdataset[sample])
	#print("Monty {} sample loaded.".format(str(sample)))
	#mfiledic[sample][0].nobg(sample)
	#print("Monty {} sample background subtracted.".format(str(sample)))
	#mfiledic[sample][0].nogvd(sample)
	#print("Monty {} sample GVD corrected.".format(str(sample)))

	i = 0
	while i < 1:
		opt = int(input("\nPlease select an option\n0. Current filelist\n1. Plotting data\n2. Global fitting\n3. New samples\n4. Saving data\n5. Exit\n"))
		if opt == 0:
			print("We have currently this sample list.")
			print(filedic)
		elif opt == 1:
			plot = [int(x) for x in input("\nPlease input sample # and plot method\n0. Single plot(before time zero)\n1. Single plot(before time zero_after GVD)\n2. Single plot(singular values)\n3. Single plot(global fit_kinetics)\n4. Single plot(global fit_spectra)\n5. Multiple plot(UV-vis)\n6. Multiple plot(spectra traces_cut)\n7. Multiple plot(spectra traces_svd)\n8. Multiple plot(spectra traces_rec)\n9. Multiple plot(kinetic traces_final)\n10. Multiple plot(kinetic traces_rec)\n11. Multiple plot(each singular vectors)\n12. Multiple plot(DAS)\n13. 2D plot(background subtraction)\n14. 2D plot(GVD correction)\n15. 2D plot(OD calibrated)\n16. 2D plot(noise subtraction)\n17. 2D plot(reconstructed)\n18. 2D plot(residual)\n19. 2D plot(final difference)\n").split()]
			plotting(filedic[plot[0]][0], plot[1])
		elif opt == 2:
			fitt = int(input("\nPlease input sample# "))
			j = 0
			while j < 1:
				print("{} sample global fitting.".format(str(fitt)))
				timeguess = [float(x) for x in input("Please enter time constants ").split()]
				filedic[fitt][0].globalfit(fitt, timeguess)
				plotting(filedic[fitt][0], 3)
				ans = input("Satisfied with the fitting?(y/n) ")
				if ans == 'y' : j += 1
				else: continue
		elif opt == 3:
			nsample = 0
			while nsample < len(newset):
				temp = nsample+24
				filedic[temp] = ['raw']
				filedic[temp][0] = maryam(temp, newset[nsample])
				print("{} sample loaded.".format(str(temp)))
				filedic[temp][0].nogvd(temp)
				print("{} sample GVD corrected.".format(str(temp)))
				filedic[temp][0].odcali(temp)
				print("{} sample OD calibrated.".format(str(temp)))

				plt.imshow(filedic[temp][0].cali, cmap='terrain', aspect="auto", clim=(-0.025, 0.02), origin='lower')
				plt.colorbar()
				plt.show()

				temp2 = np.transpose(filedic[temp][0].cali)
				plt.plot(temp2[38])
				plt.show()

				nsample += 1

		else: i = 1

gui()
