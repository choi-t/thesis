import numpy as np
import matplotlib.pyplot as plt
import h5py
import math

print("Hi, welcome to SACLA 1D XES data analysis tool.")
print("Start analyzing data...")
items=[798185]#list(range(798185,798192))+list(range(798193,798202))+list(range(798203,798206))#[798191,798193,798203,798204]#list(range(798185,798191))#list(range(798209,798221))+list(range(798227,798228))+list(range(798229,798231))+list(range(798232,798236))+list(range(798237,798239))+list(range(798241,798243))#list(range(798209,798221))

#offset=[None]*len(items)
offset1=[]
#bkgset=[]

bin50=list(range(-200,1550,200)) #-300 to 1800 or -200 to 1500
bin50set=[[] for a in range(len(bin50))]
pump50=[[] for g in range(len(bin50))]

fdelay=[9700,39700] #for dyad 2000,2700,4700,8000,14700
fdelayset=[[] for b in range(len(fdelay))]
fpump=[[] for h in range(len(fdelay))]

timingzero=950 #fixed
timezero=292 #correct t0 292
oncounts=0

for i, scan in enumerate(items):
	print("\nScan number {}".format(str(scan)))
	with h5py.File('C:/data/feco/FeKb_'+str(scan)+'.h5','r') as f: #which emission line?
		path=list(f.get('/run_'+str(scan)+'/detector_data'))
		onoff=list(f.get('/run_'+str(scan)+'/event_info/bl_3/lh_1/laser_pulse_selector_status'))
		delay=np.array(f.get('/run_'+str(scan)+'/event_info/bl_3/eh_2/eh_2_optical_delay_stage_position'))
		I1=np.array(f.get('/run_'+str(scan)+'/event_info/bl_3/eh_2/photodiode/photodiode_user_13_in_volt'))
		I0=np.array(f.get('/run_'+str(scan)+'/event_info/bl_3/eh_2/photodiode/photodiode_user_14_in_volt'))+np.array(f.get('/run_'+str(scan)+'/event_info/bl_3/eh_2/photodiode/photodiode_user_15_in_volt'))
		pump=np.array(f.get('/run_'+str(scan)+'/event_info/bl_3/eh_2/photodiode/photodiode_user_5_in_volt'))
		NDfilter=np.array(f.get('/run_'+str(scan)+'/event_info/bl_3/eh_2/eh_2_optical_ND_filter_stage_position'))
		uND, NDcount = np.unique(NDfilter, return_counts=True)
		if len(uND)==1:
			print("This scan contains one NDfilter: {}".format(str(uND)))
		else:
			print("This scan contains more NDfilters: {}".format(str(uND)))
		xfelonoff=np.array(f.get('/run_'+str(scan)+'/event_info/bl_3/eh_1/xfel_pulse_selector_status'))
		print("Detector Gain: {}".format(list(f.get('/run_'+str(scan)+'/detector_2d_1/detector_info/absolute_gain'))))
		#[1]=TMA edge, [2]=TMA edge fitting
		tma=np.loadtxt('C:/data/feco/timing/'+str(scan)+'.csv', delimiter=',', skiprows=2).transpose()[1]
		tmtag=np.loadtxt('C:/data/feco/timing/'+str(scan)+'.csv', delimiter=',', skiprows=2).transpose()[0]
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
		print("Itfy: {}".format(I1[:5]))

		#sanity check of data
		if len(path)!=len(onoff) or uxfel[0]==0:
			print("!!!Length of dataset and laser don't match, or includes xfel=0")
		elif len(xfelonoff)!=len(tma) or len(xfelonoff)!=len(NDfilter):
			print("!!!Pulse selector of xfel and/or TMA, NDfilter don't match in record-length")
		'''
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
		'''
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

			if uI0[0]!=0:
				tempset[j]=np.array(f.get('/run_'+str(scan)+'/detector_data/'+tag+'/signal'))/I0[j]
			elif cI0[j]!=0:
				tempset[j]=np.array(f.get('/run_'+str(scan)+'/detector_data/'+tag+'/signal'))/I0[j]
			else:
				tempset[j]=np.array(f.get('/run_'+str(scan)+'/detector_data/'+tag+'/signal'))

		#shot-to-shot counts on VHS detector at pixel number 350 to 425(FeKb), 140 to 220(FeKa), 80 to 120(CoKa), 140 to 240(CoKb)
		sts=np.average(np.array(tempset).transpose()[350:425], axis=0)
		print("sts Mean: {}".format(np.average(sts,axis=0)))
		#usts, csts=np.unique(sts, return_counts=True)
		#plt.plot(usts, csts)
		#plt.show()

		#Properly normalized?
		#!!!WHEN STS changes A LOT!!! -> ROI changes during the measurement...
		#or it's just because of added solvent...
		#np.savetxt('C:/data/feco/I0sts.txt',([I0,sts]))
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

		on=[[],[],[]]
		off=[]
		filtered=0

		for k, pulse in enumerate(tempset): #outliers rejection, sts 300 to 900 for FeKa, 40 to 210 for FeKb, 30 to 180 for FeKa6, 400 to 1150 for CoKa, 20 to 140 for CoKb, 150 to 450 for FeKa124,356
			#bkgset.append(pulse)
			if I0[k]<0.1 or math.isnan(tma[k]) or tma[k]>1300 or tma[k]<600 or sts[k]<40 or sts[k]>210:
				filtered+=1
			elif onoff[k]==1:
				if pump[k]!=0:
					on[0].append(tmx[k]) #corrected probe delay
					on[1].append(pulse) #tempset data
					on[2].append(pump[k]) #pump laser intensity
				else:
					print("Shot number {} is laser-on but intensity 0".format(k))
					filtered+=1
			else:
				off.append(pulse)

		print("Filtered shots: {}".format(str(filtered)))
		#offset[i]=np.average(off, axis=0)
		offset1+=off
		#print(np.std(off,axis=0))
		#plt.plot(offset[i])
		#plt.show()

		oncounts+=len(on[1])

		for e, zeit in enumerate(on[0]): #re-binning
			if -1500<=zeit<=40500: #our hard-limit of re-binning for -1ps to 19ps
				if -300<=zeit<=1500: #our hard-limit of re-binning for -1ps to 2ps
					bin50set[[round(l) for l in (bin50-zeit)/200].index(0)].append(on[1][e])
					pump50[[round(l) for l in (bin50-zeit)/200].index(0)].append(on[2][e])
				'''
				if 1500<=zeit<=2350:
					fdelayset[0].append(on[1][e])
					fpump[0].append(on[2][e])
				elif 2350<=zeit<=3200:
					fdelayset[1].append(on[1][e])
					fpump[1].append(on[2][e])
				elif 4200<=zeit<=5200:
					fdelayset[2].append(on[1][e])
					fpump[2].append(on[2][e])
				elif 7500<=zeit<=8500:
					fdelayset[3].append(on[1][e])
					fpump[3].append(on[2][e])
				elif 14200<=zeit<=15200:
					fdelayset[4].append(on[1][e])
					fpump[4].append(on[2][e])
				else:
					pass
				'''
				if 9200<=zeit<=10200:
					fdelayset[0].append(on[1][e])
					fpump[0].append(on[2][e])
				elif 39200<=zeit<=40200:
					fdelayset[1].append(on[1][e])
					fpump[1].append(on[2][e])
				else:
					pass

			else:
				pass

		'''
		for s, zeit in enumerate(tuonset[0]):
			try:
				diffset.append([zeit, (tuonset[1][s]-tuoffset[1][tuoffset[0].index(zeit)])[200]])
			except ValueError:
				pass
		'''

#print("\nbkg counts: {}".format(len(bkgset)))
#np.savetxt('C:/data/feco/CoKb_bkg.txt',(np.average(bkgset,axis=0)))
#plt.plot(np.average(bkgset,axis=0))
#plt.show()

print("\nAll sorted on-counts: {}".format(oncounts))
print("Off-counts: {}".format(len(offset1)))
offavr=np.average(offset1, axis=0)/np.trapz(np.average(offset1, axis=0))
offerror=np.std(offset1,axis=0)/np.sqrt(len(offset1))/np.trapz(np.average(offset1,axis=0))
'''
for offscan in offset:
	plt.plot(offscan/max(offscan))

plt.show()
'''
#plt.plot(offavr)
#plt.show()

print(len(fdelayset[0]))
print(len(fdelayset[1]))
avr97=np.average(fdelayset[0], axis=0)/np.trapz(np.average(fdelayset[0], axis=0))
avr97error=np.std(fdelayset[0],axis=0)/np.sqrt(len(fdelayset[0]))/np.trapz(np.average(fdelayset[0],axis=0))
avr397=np.average(fdelayset[1], axis=0)/np.trapz(np.average(fdelayset[1], axis=0))
avr397error=np.std(fdelayset[1],axis=0)/np.sqrt(len(fdelayset[1]))/np.trapz(np.average(fdelayset[1],axis=0))
np.savetxt('C:/data/feco/FeKb_dyad_798185_9.7ps_39.7ps_newnewt0_areanorm.txt', (offavr, avr97, avr397))
np.savetxt('C:/data/feco/FeKb_dyad_798185_errorbar_9.7ps_39.7ps.txt', (offerror, avr97error, avr397error, np.sqrt(avr97error**2+offerror**2), np.sqrt(avr397error**2+offerror**2)))
print(np.average(fpump[0], axis=0))
#print(np.average(fpump[1], axis=0))

plt.errorbar(list(range(len(offavr))),avr97-offavr,np.sqrt(avr97error**2+offerror**2))
#plt.errorbar(list(range(len(offavr))),avr397-offavr,np.sqrt(avr397error**2+offerror**2))
plt.show()

count50=[len(o) for o in bin50set]
#plt.plot(bin50, count50)
#plt.show()
print("count50: {}".format(count50))
'''
fcount=[len(c) for c in fdelayset]
plt.plot(fdelay, fcount)
plt.show()
print(fcount)
'''
'''
for p50 in pump50:
	plt.plot(p50)
	plt.show()
'''
avr50=[np.average(x, axis=0)/np.trapz(np.average(x, axis=0)) for x in bin50set]
avr50error=[np.std(x,axis=0)/np.sqrt(len(x))/np.trapz(np.average(x,axis=0)) for x in bin50set]
#print([np.std(x,axis=0)/np.trapz(np.average(x,axis=0)) for x in bin50set])
avr50p=[[np.average(ae,axis=0),np.std(ae,axis=0)/np.sqrt(len(ae))] for ae in pump50]

#favr=[np.average(d,axis=0)/np.trapz(np.average(d,axis=0)) for d in fdelayset]
#favrerror=[np.std(d,axis=0)/np.sqrt(len(d))/np.trapz(np.average(d,axis=0)) for d in fdelayset]
#favrp=[[np.average(m,axis=0),np.std(m,axis=0)/np.sqrt(len(m))] for m in fpump]
'''
plt.plot(bin50,avr50p)
plt.show()

for avrscan in avr50:
    plt.plot(offavr)
    plt.plot(avrscan)
    plt.show()
'''
diff50=[(s-offavr)/avr50p[r][0] for r,s in enumerate(avr50)]
diff50error=[diff50[r]*np.sqrt((s**2+offerror**2)/((avr50[r]-offavr)**2)+((avr50p[r][1]/avr50p[r][0])**2)) for r,s in enumerate(avr50error)]
#fdiff=[(q-offavr)/favrp[p][0] for p,q in enumerate(favr)]
#fdifferror=[fdiff[p]*np.sqrt((q**2+offerror**2)/((favr[p]-offavr)**2)+((favrp[p][1]/favrp[p][0])**2)) for p,q in enumerate(favrerror)]

for diff in diff50:
    plt.plot(diff)
    plt.show()
'''
for diff in fdiff:
    plt.plot(diff)
    plt.show()
'''
#np.savetxt('C:/data/feco/FeKa124_dyad_-0.3ps_to_1ps_errorbar_newnewt0_areanorm_798209-220_offerror.txt', (offerror))
#np.savetxt('C:/data/feco/FeKa124_dyad_-0.3ps_to_1ps_errorbar_newnewt0_areanorm_798209-220_onerror.txt', (avr50error))
#np.savetxt('C:/data/feco/FeKa124_dyad_-0.3ps_to_1ps_errorbar_newnewt0_areanorm_798209-220_differror.txt', (diff50error))
#np.savetxt('C:/data/feco/FeKa124_dyad_-0.3ps_to_1ps_errorbar_newnewt0_areanorm_798209-220_traceerror.txt', ([np.sqrt(np.sum(u[140:150]**2)+np.sum(u[161:175]**2)) for u in diff50error]))

#np.savetxt('C:/data/feco/FeKa124_dyad_-0.5ps_to_15ps_errorbar_newnewt0_areanorm_798191,193,203,204_offerror.txt', (offerror))
#np.savetxt('C:/data/feco/FeKa124_dyad_-0.5ps_to_15ps_errorbar_newnewt0_areanorm_798191,193,203,204_onerror.txt', (avr50error+favrerror))
#np.savetxt('C:/data/feco/FeKa124_dyad_-0.5ps_to_15ps_errorbar_newnewt0_areanorm_798191,193,203,204_differror.txt', (diff50error+fdifferror))
#np.savetxt('C:/data/feco/FeKa124_dyad_-0.5ps_to_15ps_errorbar_newnewt0_areanorm_798191,193,203,204_traceerror.txt', ([np.sqrt(np.sum(u[140:150]**2)+np.sum(u[161:175]**2)) for u in diff50error]+[np.sqrt(np.sum(t[140:150]**2)+np.sum(t[161:175]**2)) for t in fdifferror]))

#np.savetxt('C:/data/feco/FeKa124_30fs_dyad_-0.3ps_to_1.5ps_newnewt0_areanorm_798185-191,193-201,203-205.txt', ([bin50, count50, np.transpose(avr50p)[0]]))
#np.savetxt('C:/data/feco/FeKa124_30fs_dyad_-0.3ps_to_1.5ps_newnewt0_areanorm_798185-191,193-201,203-205_on.txt', (avr50))
#np.savetxt('C:/data/feco/FeKa124_30fs_dyad_-0.3ps_to_1.5ps_newnewt0_areanorm_798185-191,193-201,203-205_off.txt', (offavr))
#np.savetxt('C:/data/feco/FeKa124_30fs_dyad_-0.3ps_to_1.5ps_newnewt0_areanorm_798185-191,193-201,203-205_tr.txt', ([bin50, [np.trapz(u[140:150])+np.trapz(u[161:175]) for u in diff50]]))

#np.savetxt('C:/data/feco/FeKa356_dyad_-0.5ps_to_15ps_newnewt0_areanorm_798191,193,203,204.txt', ([bin50+fdelay, count50+fcount, list(np.array(avr50p).transpose()[0])+list(np.array(favrp).transpose()[0])]))
#np.savetxt('C:/data/feco/FeKa356_dyad_-0.5ps_to_15ps_newnewt0_areanorm_798191,193,203,204_on.txt', (avr50+favr))
#np.savetxt('C:/data/feco/FeKa356_dyad_-0.5ps_to_15ps_newnewt0_areanorm_798191,193,203,204_off.txt', (offavr))
#np.savetxt('C:/data/feco/FeKa356_dyad_-0.5ps_to_15ps_newnewt0_areanorm_798191,193,203,204_tr.txt', ([bin50+fdelay, [np.trapz(u[140:150])+np.trapz(u[161:175]) for u in diff50]+[np.trapz(af[140:150])+np.trapz(af[161:175]) for af in fdiff]]))

plt.errorbar(bin50, [np.trapz(u[360:378]) for u in diff50], [np.sqrt(np.sum(u[360:378]**2)) for u in diff50error])
plt.show()
#plt.plot(bin50, [np.trapz(u[360:378]) for u in diff50], 'bo')
#plt.show()
#plt.errorbar(bin50+fdelay, [np.trapz(u[80:104]) for u in diff50]+[np.trapz(t[80:104]) for t in fdiff], [np.sqrt(np.sum(u[80:104]**2)) for u in diff50error]+[np.sqrt(np.sum(t[80:104]**2)) for t in fdifferror])
#plt.show()
#plt.plot(bin50+fdelay, [np.trapz(u[140:150])+np.trapz(u[161:175]) for u in diff50]+[np.trapz(t[140:150])+np.trapz(t[161:175]) for t in fdiff],'b')
#plt.show()

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
