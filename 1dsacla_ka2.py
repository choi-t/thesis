import numpy as np
import matplotlib.pyplot as plt
import h5py
import math

print("Hi, welcome to SACLA 1D XES data analysis tool.")
print("Start analyzing data...")
items=list(range(739136,739160))+list(range(739181,739254))#list(range(739109,739124))+list(range(739127,739136))+list(range(739175,739181))+list(range(739254,739256))#list(range(739103,739104))+list(range(739106,739108))+list(range(739109,739124))+list(range(739127,739136))+list(range(739175,739181))+list(range(739254,739256))#list(range(739109,739124))+list(range(739127,739256))#list(range(739134,739136))+list(range(739254,739256))

#offset=[None]*len(items)
offset1=[]
#bkgset=[]

bin500=np.array(range(-1000,19500,250)) #-1ps to 19ps, 739134,135,254,255
bin500set=[[] for a in range(len(bin500))]
pump500=[[] for g in range(len(bin500))]
bin50=np.array(range(-1000,2600,50)) #-1ps to 2ps, 739109-123,127-135,175-180,254,255
bin50set=[[] for b in range(len(bin50))]
pump50=[[] for h in range(len(bin50))]
I1set=[[] for b in range(len(bin50))]
delay19=[] #19ps, 739136-159,181-253
pump19=[]

timingzero=1000 #fixed
timezero=900 #correct t0
oncounts=0

for i, scan in enumerate(items):
	print("\nScan number {}".format(str(scan)))
	with h5py.File('C:/data/sacla/ka346/Ka346_'+str(scan)+'.h5','r') as f: #which emission line?
		path=list(f.get('/run_'+str(scan)+'/detector_data'))
		onoff=list(f.get('/run_'+str(scan)+'/event_info/bl_3/lh_1/laser_pulse_selector_status'))
		delay=np.array(f.get('/run_'+str(scan)+'/event_info/bl_3/eh_2/eh_2_optical_delay_stage_position'))
		I1=np.array(f.get('/run_'+str(scan)+'/event_info/bl_3/eh_2/photodiode/photodiode_user_13_in_volt'))
		I0=np.array(f.get('/run_'+str(scan)+'/event_info/bl_3/eh_2/photodiode/photodiode_user_14_in_volt'))+np.array(f.get('/run_'+str(scan)+'/event_info/bl_3/eh_2/photodiode/photodiode_user_15_in_volt'))
		pump=np.array(f.get('/run_'+str(scan)+'/event_info/bl_3/eh_2/photodiode/photodiode_laser_1_in_volt'))
		NDfilter=np.array(f.get('/run_'+str(scan)+'/event_info/bl_3/eh_2/eh_2_optical_ND_filter_stage_position'))
		uND, NDcount = np.unique(NDfilter, return_counts=True)
		if len(uND)==1:
			print("This scan contains one NDfilter: {}".format(str(uND)))
		else:
			print("This scan contains more NDfilters: {}".format(str(uND)))
		xfelonoff=np.array(f.get('/run_'+str(scan)+'/event_info/bl_3/eh_1/xfel_pulse_selector_status'))
		print("Detector Gain: {}".format(list(f.get('/run_'+str(scan)+'/detector_2d_1/detector_info/absolute_gain'))))
		#[1]=TMA edge, [2]=TMA edge fitting
		tma=np.loadtxt('C:/data/sacla/timing/'+str(scan)+'.csv', delimiter=',', skiprows=2).transpose()[1]
		tmtag=np.loadtxt('C:/data/sacla/timing/'+str(scan)+'.csv', delimiter=',', skiprows=2).transpose()[0]
		tmx=(timingzero-tma)*2.6+(delay-timezero)*6.67
		print("TMA max, min, avrg: {}".format([max([c for c in tma if math.isnan(c)==False]), min([c for c in tma if math.isnan(c)==False]), np.average([c for c in tma if math.isnan(c)==False], axis=0)]))
		#plt.plot(tma)
		#plt.show()

		udelay, counts = np.unique(delay, return_counts=True) #how many delays?
		delayd = dict(zip(udelay, counts))
		uonoff, countss = np.unique(onoff, return_counts=True) #how many onoffs?
		onoffd = dict(zip(uonoff, countss))
		uxfel, cxfel = np.unique(xfelonoff, return_counts=True) #was there any X-ray off?
		uI1, cI1 = np.unique(I1, return_inverse=True)
		indexlist3=[ab for ab,z in enumerate(cI1) if z==0] #I1(TFY)=0 indices
		uI0, cI0 = np.unique(I0, return_inverse=True)
		indexlist4=[ac for ac,w in enumerate(cI0) if w==0] #I0=0 indices

		#sanity check of data
		if len(xfelonoff)!=len(onoff) or uxfel[0]==0:
			print("!!!Pulse selector of xfel and laser doesn't match, or includes xfel=0")
		elif len(xfelonoff)!=len(tma):
			print("!!!Pulse selector of xfel and TMA don't match in record-length")

		I1I0 = [I1[d]/c for d,c in enumerate(I0) if c!=0]
		avrI10=np.average(I1I0, axis=0)
		print("average Itfy/I0: {}".format(avrI10))

		#if max(I1/I0)>1.07*np.average(I1/I0, axis=0):
		#	print("Jet was unstable in this scan.") #Jet stability check

		if uI0[0]==0.: #I0 before sample all recorded?
			print("This scan contains I0=0.")
			print("and Index(indices): {}".format(indexlist3))
		elif uI1[0]==0.: #I1(TFY) after sample all recorded?
			print("This scan contains TFY=0.")
			print("and Index(indices): {}".format(indexlist4))
		else:
			pass

		print("Number of tags: {}".format(len(path)))
		print("Onoffs: {}".format(onoffd))
		if len(udelay)==1:
			print("Only one delay: {}".format(str(udelay)))	#One delay foot-to-foot jitter = 1.5 ps
		else:
			print("First delay: {}".format(str(delay[0])))
			print("Last delay: {}".format(str(delay[-1])))
			print("Delays: {}".format(delayd))

		### Signal-to-Noise Ratio check, rejecting bad pulses?
		#plt.plot(I0, I1, 'bo')
		#plt.show()
		#plt.plot(I1I0, 'bo') #if jet was fluctuating, excited fraction might differ as well...
		#plt.show()

		tempset=[None]*len(path)
		for j, tag in enumerate(path): #raw data into tempset with normalization
			if int(tmtag[j])!=int(tag[4:]):
				print("tmtag is not same with tag of data")
				print("tmtag: {}".format(str(tmtag[j])))
				print("tag: {}".format(str(tag)))

			if uI1[0]!=0.:
				tempset[j]=np.array(f.get('/run_'+str(scan)+'/detector_data/'+tag+'/signal'))/I1[j]
			elif cI1[j]!=0:
				tempset[j]=np.array(f.get('/run_'+str(scan)+'/detector_data/'+tag+'/signal'))/I1[j]
			else: #this is NaN anyway... exclude it later
				tempset[j]=np.array(f.get('/run_'+str(scan)+'/detector_data/'+tag+'/signal'))

		#shot-to-shot counts on VHS detector at pixel number 120 to 220
		sts=np.average(np.array(tempset).transpose()[120:220], axis=0)
		print("sts Mean: {}".format(np.average(sts,axis=0)))
		#usts, csts=np.unique(sts, return_counts=True)
		#plt.plot(usts, csts)
		#plt.show()

		#Properly normalized?
		#plt.plot(I0, sts, 'bo')
		#plt.show()
		#plt.plot(I1, sts, 'bo')
		#plt.show()

		#corrected probe delay distribution for re-binning
		#utmx, ctmx=np.unique(tmx, return_counts=True)
		#plt.plot(utmx, ctmx)
		#plt.show()

		#time sorting
		if max([c for c in tma if math.isnan(c)==False])>1800 or min([c for c in tma if math.isnan(c)==False])<300:
			print("Timing Feedback was partially out in this scan.")
			plt.plot(tma)
			plt.show()

		on=[[],[],[],[]]
		off=[]
		filtered=0

		for k, pulse in enumerate(tempset): #outliers rejection, sts 3000 to 25000 for Kb, 60 to 1500 for VtC, 5000 to 10000 for Ka246, 2000 to 7000 for Ka135, 4500 to 8000 for Ka125 and Ka346
			if I1[k]<0.01 or math.isnan(tma[k]) or tma[k]>1300 or tma[k]<600 or I1[k]/I0[k]<avrI10-0.01 or I1[k]/I0[k]>avrI10+0.01 or sts[k]<4500 or sts[k]>8000:
				filtered+=1
			elif onoff[k]==1:
				if pump[k]!=0:
					on[0].append(tmx[k]) #corrected probe delay
					on[1].append(pulse) #tempset data
					on[2].append(pump[k]) #pump laser intensity
					on[3].append(I1[k]/I0[k])
				else:
					print("Shot number {} is laser-on but intensity 0".format(k))
					filtered+=1
			else:
				off.append(pulse)

		print("Filtered shots: {}".format(str(filtered)))
		#offset[i]=np.average(off, axis=0)
		offset1+=off
		#plt.plot(offset[i])
		#plt.show()

		oncounts+=len(on[1])
		#test19=[] #really bad
		#temp50set=[[] for b in range(len(bin50))] #really bad

		for e, zeit in enumerate(on[0]): #re-binning
			if -1125<=zeit<=19375: #our hard-limit of re-binning for -1ps to 19ps
				if -1025<=zeit<=2575: #our hard-limit of re-binning for -1ps to 2ps
					bin50set[[round(l) for l in (bin50-zeit)/50].index(0)].append(on[1][e])
					pump50[[round(l) for l in (bin50-zeit)/50].index(0)].append(on[2][e])
					I1set[[round(l) for l in (bin50-zeit)/50].index(0)].append(on[3][e])

				if 18500<=zeit:
					delay19.append(on[1][e])
					pump19.append(on[2][e])

				bin500set[[round(m) for m in (bin500-zeit)/250].index(0)].append(on[1][e])
				pump500[[round(m) for m in (bin500-zeit)/250].index(0)].append(on[2][e])
			else:
				pass

		#delay19.append(np.average(test19,axis=0)) #really bad
		#for s, temp in enumerate(temp50set): #really bad
		#	if len(temp)!=0:
		#	    bin50set[s].append(np.average(temp, axis=0))
		'''
		for s, zeit in enumerate(tuonset[0]):
			try:
				diffset.append([zeit, (tuonset[1][s]-tuoffset[1][tuoffset[0].index(zeit)])[200]])
			except ValueError:
				pass
		'''

print("\nAll sorted on-counts: {}".format(oncounts))
print("On-counts at 19ps: {}".format(len(delay19)))
print("Off-counts: {}".format(len(offset1)))
offavr=np.average(offset1, axis=0)/np.trapz(np.average(offset1, axis=0))
offerror=np.std(offset1,axis=0)/np.sqrt(len(offset1))/np.trapz(np.average(offset1, axis=0))

'''
for offscan in offset:
	plt.plot(offscan)

plt.show()

avr19=np.average(delay19, axis=0) #19ps data average
avr19p=np.average(pump19, axis=0)
print(avr19p)
#plt.plot(offavr)
#plt.plot(avr19)
#plt.show()
plt.plot(avr19-offavr)
plt.show()
#np.savetxt('C:/data/sacla/absolute_19ps_Ka346_Ipump0.2518_739136-159,181-253.txt', (offavr, avr19))
'''
#count500=[len(n) for n in bin500set] #counts at each delay point of kinetics
count50=[len(o) for o in bin50set]

#plt.plot(bin500, count500)
#plt.show()
plt.plot(bin50, count50)
plt.show()
'''
for p500 in pump500:
	plt.plot(p500)
	plt.show()

for p50 in pump50:
	plt.plot(p50)
	plt.show()
'''
#avr500=[np.average(v,axis=0)/np.trapz(np.average(v,axis=0)) for v in bin500set] #average of kinetics
#avr500error=[np.std(v,axis=0)/np.sqrt(len(v))/np.trapz(np.average(v,axis=0)) for v in bin500set]
avr50=[np.average(x,axis=0)/np.trapz(np.average(x,axis=0)) for x in bin50set]
avr50error=[np.std(x,axis=0)/np.sqrt(len(x))/np.trapz(np.average(x,axis=0)) for x in bin50set]
#avrI1=[np.average(x,axis=0) for x in I1set]
#avr500p=[[np.average(ad,axis=0),np.std(ad,axis=0)/np.sqrt(len(ad))] for ad in pump500]
#avr500p=[np.average(ad,axis=0) for ad in pump500]
#avr50p=[np.average(ae,axis=0) for ae in pump50]
avr50p=[[np.average(ae,axis=0),np.std(ae,axis=0)/np.sqrt(len(ae))] for ae in pump50]
'''
plt.plot(bin500,avr500p) #fluctuation of pump laser intensity
plt.show()
plt.plot(bin50,avr50p)
plt.show()

for avrscan in avr50:
    plt.plot(offavr)
    plt.plot(avrscan)
    plt.show()
'''
#diff500=[(q-offavr)/avr500p[p][0] for p,q in enumerate(avr500)]
#diff500error=[diff500[p]*np.sqrt((q**2+offerror**2)/((avr500[p]-offavr)**2)+((avr500p[p][1]/avr500p[p][0])**2)) for p,q in enumerate(avr500error)]
diff50=[(s-offavr)/avr50p[r][0] for r,s in enumerate(avr50)]
diff50error=[diff50[r]*np.sqrt((s**2+offerror**2)/((avr50[r]-offavr)**2)+((avr50p[r][1]/avr50p[r][0])**2)) for r,s in enumerate(avr50error)]
'''
for diff in diff50:
    plt.plot(diff)
    plt.show()
'''
#np.savetxt('C:/data/sacla/-1ps_to_20ps_errorbar_ka346_250fs_areanorm_739134,135,254,255_offerror.txt', (offerror))
#np.savetxt('C:/data/sacla/-1ps_to_20ps_errorbar_ka346_250fs_areanorm_739134,135,254,255_onerror.txt', (avr500error))
#np.savetxt('C:/data/sacla/-1ps_to_20ps_errorbar_ka346_250fs_areanorm_739134,135,254,255_differror.txt', (diff500error))
#np.savetxt('C:/data/sacla/-1ps_to_20ps_errorbar_ka346_250fs_areanorm_739134,135,254,255_traceerror.txt', ([np.sqrt(np.sum(u[150:180]**2)+np.sum(u[205:220]**2)) for u in diff500error]))

#np.savetxt('C:/data/sacla/-1ps_to_2ps_Cudmp_Ka125_250fs_739109-123,127-135,175-180,254,255_absolute.txt', ([bin50, count50, avr50p]))
#np.savetxt('C:/data/sacla/-1ps_to_2ps_Cudmp_Ka125_250fs_739109-123,127-135,175-180,254,255_absolute_on.txt', (avr50))
#np.savetxt('C:/data/sacla/-1ps_to_2ps_Cudmp_Ka125_250fs_739109-123,127-135,175-180,254,255_absolute_off.txt', (offavr))
#np.savetxt('C:/data/sacla/-1ps_to_2ps_Cudmp_Ka125_250fs_739109-123,127-135,175-180,254,255_absolute_tr.txt', ([bin50, [np.trapz(af[150:180])+np.trapz(af[205:220]) for af in diff50]]))

#np.savetxt('C:/data/sacla/-1ps_to_20ps_Cudmp_Ka346_250fs_739134,135,254,255_absolute.txt', ([bin500, count500, avr500p]))
#np.savetxt('C:/data/sacla/-1ps_to_20ps_Cudmp_Ka346_250fs_739134,135,254,255_absolute_on.txt', (avr500))
#np.savetxt('C:/data/sacla/-1ps_to_20ps_Cudmp_Ka346_250fs_739134,135,254,255_absolute_off.txt', (offavr))
#np.savetxt('C:/data/sacla/-1ps_to_20ps_Cudmp_Ka346_250fs_739134,135,254,255_absolute_tr.txt', ([bin500, [np.trapz(af[150:180])+np.trapz(af[205:220]) for af in diff500]]))

#plt.plot(bin50, avrI1, 'b')
#plt.show()
#plt.plot(bin500, [np.trapz(t[150:180])+np.trapz(t[205:220]) for t in diff500], 'b')
#plt.show()
#plt.plot(bin500, [np.trapz(u[150:180]) for u in diff500], 'b')
#plt.plot(bin500, [np.trapz(u[205:220]) for u in diff500], 'r')
#plt.show()
plt.errorbar(bin50, [np.trapz(u[150:180])+np.trapz(u[205:220]) for u in diff50], [np.sqrt(np.sum(u[150:180]**2)+np.sum(u[205:220]**2)) for u in diff50error])
plt.show()

'''
def gui():
	y = 0
	while y < 1:
		opt = int(input("\nPlease select an option\n0. All scan numbers\n1. \n2. Global fitting\n3. New samples\n4. Saving data\n5. Exit\n"))
		if opt == 0:
			print(items)
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

		else: y = 1

gui()


if len(dset[0])==2:
	print("All on-counts: {}".format(oncounts))
	avrdset=np.average(dset, axis=0)
	plt.plot(avrdset[0])
	plt.plot(avrdset[1])
	plt.show()
	plt.plot((avrdset[1]-avrdset[0]))
	plt.show()

	#np.savetxt('C:/data/sacla/delay1600_Cudmp_Kb.txt', (avrdset))
else:
	for offscan in offset:
		plt.plot(offscan)

	plt.show()
	toff = np.average(offset, axis=0)
	totalbin = np.sum([e[0] for e in dset], axis=0)

	tonn = np.average([u[1] for u in dset], axis=0)
	tpump = np.average([u[2] for u in dset], axis=0)

	diffset=[(v-toff)/tpump[s] for s,v in enumerate(tonn)]


	tempon=None
	temppump=None
	tonn=[]
	tpump=[]

	for u in dset:
		if u[0][-1]!=0:
			tonn.append(u[1])
			tpump.append(u[2])
		else:
			tempon=u[1]
			temppump=u[2]
			print("this notice should appear only once")

	tonnavr=np.average(tonn,axis=0)
	tpumpavr=np.average(tpump,axis=0)

	short=np.average([tonnavr[:50], tempon[:50]], axis=0)
	shortpump=np.average([tpumpavr[:50], temppump[:50]], axis=0)
	shortdiff=[(v-toff)/shortpump[s] for s,v in enumerate(short)]
	longdiff=[(g-toff)/tpumpavr[50+h] for h,g in enumerate(tonnavr[50:])]

	diffset=shortdiff+longdiff


	plt.plot(delayx, totalbin) #re-binned delay vs. counts
	plt.show()
	plt.plot(delayx, [np.trapz(t[180:215]) for t in diffset], 'bo')
	plt.show()

	#for scan in diffset:
	#	plt.plot(scan)
	#	plt.show()

'''
