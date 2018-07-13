#########################################
###   ##   ### ######   ###     #     ###
## ##### ## ## ##### ### #### ##### #####
### ####   ### ##### ### #### ##### #####
#### ### ##### ##### ### #### ##### #####
#   #### #####    ###   ##### ##### #####
#########################################

#########################################
### SPPlot ############## Version: 140916
#########################################
#################### Author: Jack Morford
#########################################

### A script to create multiple plot files - intended to recreate AIPS' SPFLG plotting function for Radio data stored in AIPS

### A python script writen in Parseltongue for formulation within AIPS

### This script 'should' not need altering, simply edit the input file named: SPPlot_input.py

### The core plotting routines have been adapted from PLOT_LIB routines written by Hans-Rainer Klockner.

### Updates:
# Oct 2014 - Version 1.0: original (J.Morford)
# Feb 2015 - Version 1.1: major upgrades, parallelised for machines with multiple CPUs using pythons multiprocessing module (J.Morford)
# Apr 2016 - Version 1.2: addition of combined plots (thanks go to Javier Moldon for this idea and piece of code), option to plot
#			 specific timerange (J.Morford), option to read in flag table attached to data (original function written by
#			 DMF for use within SERPent, adapted by J.Morford to work within SPPlot).
# May 2016 - Version 1.3: re-writing of visappend function (now visaapend2) - now only reads the fits file once to save memory issues
#			 for those using smaller machines whilst considerably speeding up the programme, changes made by J.Moldon. Further
# 			 minor changes made by J.Moldon - see github commits for details.
# Aug 2016 - Version 1.3.1: visappend function again re-written to vissappend3 - now calculates free memory vs spplot memory usage
# 			 and will change the way it handles the data accordingly. If many files need to be opened at once (> 1016) it will give the
# 			 user the option to continue by reading all the arrays into the memory, else it'll save each array to a file.
# Sep 2-16 - Version 1.3.2: re-worked the visappend function - now uses PyTables to create two files, one for the times arrays the other
#			 for the visibility arrays. Each file contains a separate Earray for each IF, BSL, ST, TimeRange - this means only small
# 			 amounts of data are read into the memory at any one time no matter how many arrays one needs to create.

### Intructions for use are found in the input file - email jmorford@star.ucl.ac.uk for bugs/issues/questions

# Type " parseltongue SPPlot.py --version " - for version information.

#########################################
#########################################
### Modules #############################

import os
import sys
import cPickle as pkle
import os.path
import multiprocessing as mp
import collections

from AIPS import AIPS, AIPSDisk
from AIPSTask import AIPSTask, AIPSList
from AIPSData import AIPSUVData, AIPSImage
import Wizardry.AIPSData
from Wizardry.AIPSData import AIPSUVData as WizAIPSUVData

import numpy as np
from numpy import std,mean,median,arange, arctan2

import matplotlib
matplotlib.use('Agg') # force the antigrain backend
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import figure, close, savefig, rcParams, cm, cla, clf, text
from matplotlib.colors import LogNorm
from matplotlib._png import read_png
from pylab import *

import tables

import math, time, datetime
from time import gmtime, strftime, localtime
ti = time.time()    # To time the script


#########################################
#########################################
### Functions ###########################

def time2hms(seconds):
    h=int(seconds/3600)
    m=int(seconds % 3600)/60
    s=seconds-(h*3600)-(m*60)
    h=`h`
    m=`m`
    s="%4.2f" % s
    hms=h.zfill(2)+":"+m.zfill(2)+":"+s.zfill(4)
    return hms

def array_size(data_mem):
    ## Returns the size of the array with the appropriate suffix
    if len(str(data_mem)) <= 6:
        return "%.3f KB" % (data_mem / 10.0**3)
    elif len(str(data_mem)) <= 9:
        return "%.3f MB" % (data_mem / 10.0**6)
    elif len(str(data_mem)) <= 12:
        return "%.3f GB" % (data_mem / 10.0**9)
    elif len(str(data_mem)) <= 15:
        return "%.3f TB" % (data_mem / 10.0**12)


def computer_memory(path2folder):
    import os
    memfile = open(path2folder + 'mem_stats.txt', 'wr')
    memfile.flush()
    file = path2folder + 'mem_stats.txt'
    os.system('free -k >' + file)    # need to test this!
    #os.system('free -k > mem_stats.txt')    # need to put in file path
    memfile = open(path2folder + 'mem_stats.txt', 'r')
    stats = []
    for line in memfile:
        string = ''
        for char in xrange(len(line)):
            if line[char].isdigit():
                string += line[char]
                if line[char+1] == ' ' or line[char+1] == '\n':
                    stats.append(int(string))
                    string = ''
    global mem_stats
    mem_stats = {}
    mem_stats['mem_total'] = stats[0]
    mem_stats['mem_used'] = stats[1]
    mem_stats['mem_free'] = stats[2]
    mem_stats['mem_shared'] = stats[3]
    mem_stats['mem_buffers'] = stats[4]
    mem_stats['mem_cached'] = stats[5]
    mem_stats['buf/cach_used'] = stats[6]
    mem_stats['buf/cach_free'] = stats[7]
    mem_stats['swap_total'] = stats[8]
    mem_stats['swap_used'] = stats[9]
    mem_stats['swap_free'] = stats[10]
    memfile.close()
    return mem_stats


def mergepdfs(listtomerge):

	mergedfile = outfilename+'.pdf'

	if len(listtomerge) > 1 and len(listtomerge) < 3:
		first = str(listtomerge[1])
		second = str(listtomerge[0])
		os.system('gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile='+str(path2folder+mergedfile)+' '+str(path2folder+first)+' '+str(path2folder+second))

		os.remove(path2folder+first)
		os.remove(path2folder+second)

		print "\nMerge Completed --> "+str(path2folder+mergedfile)

	if len(listtomerge) > 2:
		listtomerge2 = []
		for i in xrange(len(listtomerge)):
			listtomerge2.append(str(path2folder)+listtomerge[i])

			#listtomerge2 = listtomerge2[::-1] # this just reverses the list, puts the pages in reverse order

			mystring2 = ' '.join(listtomerge2)

		#print "MYSTRING2:",mystring2
		os.system('gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile='+str(path2folder+mergedfile)+' '+str(mystring2))

		for i in xrange(len(listtomerge)):
			os.remove(path2folder+listtomerge[i])

		print "\nMerge Completed --> "+str(path2folder+mergedfile)

	if len(listtomerge) == 1:
		print "\nNo merging required - only one plotfile"
		print "File saved at --> ", str(listtomerge[0])


def getpagetimelist(ntimescans,timeperpage):

	pagetimelist = [0]
	numofpages = (ntimescans/timeperpage)

	for i in xrange(1,numofpages+1):
		pagetimelist.append(timeperpage*i)

	return pagetimelist

def time24(t):

	from math import floor

	day = floor(t)

	hour = 24*(t-floor(t))
	min  = (hour - floor(hour))*60
	sec  = round((min - floor(min))*60,0)

	#return str(str(int(day))+':'+str(int(hour))+':'+str(int(min))+':'+str(int(sec)))
	return '{0}/{1:02d}:{2:02d}:{3:02d}'.format(int(day), int(hour), int(min), int(sec))


def hms2dectime(time):

	tmp = time.split(" ")
	day = int(tmp[0])
	hour = float(int(tmp[1])/24.)
	minute = float(int(tmp[2])/(60.*24))
	sec = float(int(tmp[3])/(3600.*24))

	digital = sec + minute + hour + float(day)

	return digital


def getyticks(wizdata,timesarray, sourceidlist, yaxisticks, ticksperpage):

	timesarray = [float(i) for i in timesarray]
	timesarray = np.array(timesarray)

	timestart = []
	timeend = []
	nx = wizdata.table('NX',1)
	for i in nx:
		if i['source_id'] in sourceidlist:
			nxstarttime = i['time']
			intervaltime = i['time_interval']
			starttime = nxstarttime-(0.5*(intervaltime))
			timestart.append(starttime)

	indexlist = []
	count = 6
	x = len(timestart)
	while x > 0:
		for i in timestart:
			y = round(i,count)
			if y in timesarray:
				index = np.where(timesarray==y)
				if index[0][0]+1 not in indexlist:
					indexlist.append(index[0][0]+1)
					x -= 1
		count += 1
		if count > 20:
			break

	indexlist.append(len(timesarray)-1)

	if 1 not in indexlist:
		indexlist.append(1)

	for n,i in enumerate(indexlist):
		if i == 1:
			indexlist[n] = 0

	indexlist.sort()

	labellist = []
	marker = 0
	for i in indexlist:
		if marker == 0 and len(indexlist) > 1:
			labellist.append(timesarray[i])
		else:
			labellist.append(timesarray[i-1])
		if marker > 0 and marker < len(indexlist)-1:
			labellist.append(timesarray[i-2])
		if marker == len(indexlist)-1:
			labellist.append(timesarray[i-1])
		marker += 1
	labellist.sort()


	tlabels = []
	for i in xrange(1,len(labellist),2):
		tmp = []
		tmp2 = []
		a = time24(float(labellist[i-1]))
		b = time24(float(labellist[i]))
		c = str(a)+' - '+str(b)

		tlabels.append(c)

	tlabels.append(time24(float(labellist[-1])))

	yscans = indexlist

	if yaxisticks == 'tickperpage' and len(timesarray) > ticksperpage:
		timeticklist = []
		indexlist = []

		for time in xrange(0, len(timesarray), len(timesarray)/ticksperpage):
			timeticklist.append(timesarray[time])
			indexlist.append(time)

		timeticklist.append(timesarray[len(timesarray)-1])
		indexlist.append(len(timesarray-1))

		if len(timeticklist) > (1+ticksperpage):
			timeticklist.pop()
			indexlist.pop(ticksperpage)

		tlabels = []
		for time in timeticklist:
			tlabels.append(time24(float(time)))



	return indexlist, tlabels, yscans


def get_ordered_pols(uvdata):
    ### Function to determine the AIPS flag entry appropriate index for
    # the present stokes parameters.

    ## Figure out number of Stokes and polarizations
    npol = len(uvdata.stokes)
    # print " Number of Polarizations (npol): %i" % npol
    orderedpols = {}
    for x in xrange(npol):
        if uvdata.stokes[x] == 'RR' or uvdata.stokes[x] == 'XX':
            orderedpols[str(uvdata.stokes[x])] = 0
        elif uvdata.stokes[x] == 'LL' or uvdata.stokes[x] == 'YY':
            orderedpols[str(uvdata.stokes[x])] = 1
        elif uvdata.stokes[x] == 'RL' or uvdata.stokes[x] == 'XY':
            orderedpols[str(uvdata.stokes[x])] = 2
        elif uvdata.stokes[x] == 'LR' or uvdata.stokes[x] == 'YX':
            orderedpols[str(uvdata.stokes[x])] = 3
    return orderedpols


def sortcomb_pages(ntimescans,combimagelist):

	pagedictionary = collections.OrderedDict()

	for bsl,ntimescan in ntimescans.items():
		timeranglist = []
		pagetimelist = getpagetimelist(ntimescan,timeperpage)

		for starttime in pagetimelist:
			visstart = starttime
			visend = starttime+(timeperpage-1)
			if visend > ntimescan:
				visend = ntimescan

			if visend - visstart == (timeperpage-1):
				timerang = str(visstart)+'-'+str(visend)
				if timerang not in timeranglist:
					timeranglist.append(timerang)

	page = 1
	lastpage = len(timeranglist)+1
	lastpagelist = []
	allbutlastpagelist = []

	for timerang in timeranglist:
		tmplist = []
		for image in combimagelist:
			if timerang in image:
				tmplist.append(image)
				allbutlastpagelist.append(image)
			pagedictionary[page] = tmplist

		page += 1

	for image in combimagelist:
		if image not in allbutlastpagelist:
			lastpagelist.append(image)

	timeranglist.append((str(pagetimelist[-1])+'-end'))

	pagedictionary[lastpage] = lastpagelist

	return pagedictionary, timeranglist

##### generate custom colormap

import matplotlib.colors as col

def anglecolormap():
    # Generate colormap for angles (180 = -180).
    #white = '#ffffff'
    #black = '#000000'
    #red = '#ff0000'
    #green = '#00ff00'
    blue = '#5555dd'
    black = '#555555'
    red = '#ebc5c5'
    green = '#81cc81'
    anglemap = col.LinearSegmentedColormap.from_list(
        'anglemap', [black, green, red,  black], N=256, gamma=1)
    return anglemap

anglemap = anglecolormap()

##########################################################################
# Load in a flag table multiprocess function - original by Danielle Fenech
# editted by Jack Morford for use within SPPlot...
##########################################################################

# Parrallel function to load in a flag array per baseline, per timeperpage - save the numpy arrays in path2folder in same format as visappend function
def load_in_flags(send_q, rec_q, cpu):

	# This function needs:
	# uvdata, flagtablelist, jobnmber, totalnumofjobs, sourcelist, bline, orderedpols, nchans, stokes, ntimescans, times_array, visstart, visend, noIFs, path2folder

	for value in iter(send_q.get, 'STOP'):

		time0 = time.time()
		Name, Klass, Disk, Seq, times_array, flag_tables, sourceidlist, bline, orderedpols, indxorderedpol, stokes, nchans, noIFs, npol, visstart, visend, path2folder, jobnum, totaljobs = value

		out_val = int(1)
		ntimescans = len(times_array)

		uvdata = WizAIPSUVData(Name, Klass, Disk, Seq)

		old_flags = np.zeros([noIFs,npol,ntimescans,nchan],dtype=np.int8)

		JOB = 'Job %i of %i: ' % (jobnum, totaljobs)

		for fg in flag_tables:
			fgtab = uvdata.table('FG',fg)

			print JOB+'Loading in previous flag table %i for baseline %s' % (fg, bline)

			for row in fgtab:
				#print sys.getsizeof(row)
				if row.source in sourceidlist or row.source == 0:
					if (row.ants[0] == int(bline[0]) or row.ants[0] == 0) and (row.ants[1] == int(bline[-1]) or row.ants[1] == 0):

						tempol = np.array(row.pflags)
						tempol = tempol[np.array(np.sort(orderedpols.values()))]
						polarr = np.where(tempol>0)

						schan = row.chans[0]-1
						if row.chans[1] == 0:
							echan = nchan
						else:
							echan = row.chans[1]

						sif = row.ifs[0]-1
						if row.ifs[1] == 0:
							eif = nif
						else:
							eif = row.ifs[1]

						stime = row.time_range[0]
						etime = row.time_range[1]
						temptime = np.where((times_array>=stime) & (times_array<=etime))

						for pol in polarr[0]:
							old_flags[sif:eif,pol,temptime,schan:echan] = out_val

		time2 = time.time()
		print JOB+'Creating flags on CPU', cpu, 'took:', time2hms(time2-time0)

		# Split each old_flags file into further divisions, i.e. per stoke and per IF
		# this ensures when they're read back in, they are in the same format/shape as
		# the amplitude arrays in the vissapend function

		numpysavelist = []
		for IF in xrange(len(old_flags)):
			for key, value in indxorderedpol.items():
				stname = key
				array = old_flags[IF][value]

				numpyname = path2folder+str(bline)+"_"+str(stname)+"_"+str(visstart)+"-"+str(visend)+'_'+str(IF+1)+'_flags'
				np.save(numpyname, array)
				del array
				numpysavelist.append(numpyname)

		del old_flags
		del times_array
		del bline
		del sourceidlist
		del noIFs
		del orderedpols

		numpynamedict = {jobnum: numpysavelist}
		rec_q.put(numpynamedict)


########################################################################################
### Visappend function - another parrallel process to read in the actual visibility data
### and store them into arrays per baseline,timeperpage,st,IF. It will then pickle this
### array and delete it from the memory so it can be read in later...
########################################################################################

def visappend(s_q, r_q, cpu):

	for value in iter(s_q.get, 'STOP'):

		Name,Klass,Disk,Seq,visstart,visend,baseline,IF,polnames,st,nchan,sourceidlist,jobnumber,visnumarray=value[:]

		wizdata = WizAIPSUVData(Name, Klass, Disk, Seq)

		arraylength = visend-visstart

		times_array = np.zeros(arraylength, dtype='|S12')
		vis_array = np.zeros(shape=(arraylength,nchan))

		appending_time = 0
		appending_count = 0
		vis_count = 0

		print "---> Appending Visibilities on CPU: {0:2d}".format(cpu+1),"for Bsl:",baseline, "Stoke:",st, "IF:",IF+1, "Timerange:",visstart,"-",visend

		if choosesources == 'all':
			for vis in visnumarray:
				curvisarray = wizdata[vis].visibility[IF]

				times_array[appending_time] = round(float(wizdata[vis].time),6)
				appending_time += 1

				if amporphas == 'P':
					vis_array[appending_count, 0:(nchan)] = np.where(curvisarray[:,polnames[st]][:,2] > 0.0, (360./(2*pi))*arctan2(curvisarray[:,polnames[st]][:,1], curvisarray[:,polnames[st]][:,0]), float('NaN'))
					appending_count += 1

				if amporphas == 'A':
					vis_array[appending_count, 0:(nchan)] = np.where(curvisarray[:,polnames[st]][:,2] > 0.0, (np.sqrt((curvisarray[:,polnames[st]][:,0])**2 + (curvisarray[:,polnames[st]][:,1])**2)), float('NaN'))
					appending_count += 1

		if choosesources == 'choose':
			for vis in visnumarray:
				curvisarray = wizdata[vis].visibility[IF]
				try:
					sourceidnum = wizdata[vis].source
					if sourceidnum in sourceidlist:
						times_array[appending_time] = round(float(wizdata[vis].time),6)
						appending_time += 1

					if amporphas == 'P':
						vis_array[appending_count, 0:(nchan)] = np.where(curvisarray[:,polnames[st]][:,2] > 0.0, (360./(2*pi))*arctan2(curvisarray[:,polnames[st]][:,1], curvisarray[:,polnames[st]][:,0]), float('NaN'))
						appending_count += 1

					if amporphas == 'A':
						vis_array[appending_count, 0:(nchan)] = np.where(curvisarray[:,polnames[st]][:,2] > 0.0, (np.sqrt((curvisarray[:,polnames[st]][:,0])**2 + (curvisarray[:,polnames[st]][:,1])**2)), float('NaN'))
						appending_count += 1

				except KeyError:
					times_array[appending_time] = round(float(wizdata[vis].time),6)
					appending_time += 1

					if amporphas == 'P':
						vis_array[appending_count, 0:(nchan)] = np.where(curvisarray[:,polnames[st]][:,2] > 0.0, (360./(2*pi))*arctan2(curvisarray[:,polnames[st]][:,1], curvisarray[:,polnames[st]][:,0]), float('NaN'))
						appending_count += 1

					if amporphas == 'A':
						vis_array[appending_count, 0:(nchan)] = np.where(curvisarray[:,polnames[st]][:,2] > 0.0, (np.sqrt((curvisarray[:,polnames[st]][:,0])**2 + (curvisarray[:,polnames[st]][:,1])**2)), float('NaN'))
						appending_count += 1


		if do_loadflag == 'yes':
			flagfile = str(baseline)+"_"+str(st)+"_"+str(visstart)+"-"+str(visend)+'_'+str(IF+1)+'_flags.npy'
			flags = np.load(path2folder+flagfile)

			vis_array[np.where(flags > 0)] = float('NaN')

			del flags
			os.remove(path2folder+flagfile)

		picklevisarrayname = str(baseline)+"_"+str(st)+"_"+str(visstart)+"-"+str(visend)+'_'+str(IF+1)+".p"
		picklevisarray = picklepath+'/'+picklevisarrayname
		pickletimesarrayname = 'timesarray_'+str(baseline)+'_'+str(st)+'_'+str(visstart)+"-"+str(visend)+'.p'
		pickletimesarray = picklepath+'/'+pickletimesarrayname

		pkle.dump(vis_array, open(picklevisarray, "wb"))
		pkle.dump(times_array, open(pickletimesarray, "wb"))

		picklenamedict = {jobnumber: str(picklevisarrayname)}

		del vis_array
		del curvisarray
		del times_array

		r_q.put(picklenamedict)

def visappend2():

    # Find visstart and visend:
    timeblockdict = {}
    for bline, totntimescans in ntimescans.items():
	    print bline, totntimescans
	    totntimescans = int(totntimescans)

	    if timeperpage >= totntimescans:
	    	timeblock = []
	        visstart = 0
	        visend = totntimescans
	        timeblock.append([visstart, visend])

	    if timeperpage < totntimescans:
	        timeblock = []
	        pagetimelist = getpagetimelist(totntimescans,timeperpage)
	        for starttime in pagetimelist:
	            visstart = starttime
	            visend = starttime+(timeperpage-1)
	            if visend > totntimescans:
	               visend = totntimescans
	            timeblock.append([visstart, visend])

	    timeblockdict[bline] = timeblock

    print "\nTIMEBLOCKDICT:", timeblockdict

    file_count = 0
    # Initialize arrays and output files
    vis_array = {}
    vis_output = {}
    times_output = {}
    time_count = {}   # Number of times that have been appended. Individual counter for baseline, st, IF
    current = {}      # Current time block
    print '\nReading labels and naming output files'
    #for baseline in baselines:
    for baseline, timeblock in timeblockdict.items():

        for tblock in timeblock:
            labeltime = '{}_{}-{}'.format(baseline, tblock[0], tblock[1])
            timesarrayname = 'timesarray_'+labeltime+'.npy'
            try:
                os.remove(picklepath+'/'+timesarrayname)
            except:
                pass
            times_output[labeltime] = open(picklepath+'/'+timesarrayname, "ab")
            file_count += 1

            print file_count

            for st in stokes:
                for IF in xrange(IF_start-1,IF_end):
                    label = '{}_{}_{}-{}_{}'.format(baseline, st, tblock[0], tblock[1], IF+1)
                    # Arrays
                    vis_array[label] = np.zeros(shape=(nchan))
                    # Output files
                    visarrayname = label+'.npy'
                    try:
                        os.remove(picklepath+'/'+visarrayname)
                    except:
                        pass
                    vis_output[label] = open(picklepath+'/'+visarrayname, "ab")
                    file_count += 1
                    print file_count
                    #vis_output[label] = picklepath+'/'+visarrayname

    print current
    print time_count
    #print "VIS_ARRAY", vis_array
    #print "VIS_OUTPUT", vis_output
    #print "TIMES_OUTPUT", times_output


    # Iterate ove/finalspplots/pir all visibilities and write to corresponding output file
    print 'Populating vis_arrays'

    for row in wizdata:
        baseline = '{}-{}'.format(row.baseline[0], row.baseline[1])
    	current[baseline] = 0
    	time_count[baseline] = 0
        if baseline != '1-2':
            if (row.source in sourceidlist) and (baseline in baselines):
                curvisarray = row.visibility
                #tblock = timeblock[current[baseline]]
                #print "current[baseline]", current[baseline]
                #print "\ntblock", timeblock[current[baseline]]
                tblock = timeblockdict[baseline][current[baseline]]

                labeltime = '{}_{}-{}'.format(baseline, tblock[0], tblock[1])
                #print labeltime

                print time_count[baseline]
                if (time_count[baseline] >= tblock[1]):
                    current[baseline] += 1

                np.save(times_output[labeltime], round(float(row.time),6))
                time_count[baseline] += 1

                for st in stokes:
                    for IF in xrange(IF_start-1,IF_end):
                        label = '{}_{}_{}-{}_{}'.format(baseline, st, tblock[0], tblock[1], IF+1)
                        if amporphas == 'P':
                            vis_array[label] = np.where(curvisarray[IF,:,polnames[st]][:,2] > 0.0, (360./(2*pi))*arctan2(curvisarray[IF,:,polnames[st]][:,1], curvisarray[IF,:,polnames[st]][:,0]), float('NaN'))
                        if amporphas == 'A':
                            vis_array[label] = np.where(curvisarray[IF,:,polnames[st]][:,2] > 0.0,(np.sqrt((curvisarray[IF,:,polnames[st]][:,0])**2 + (curvisarray[IF,:,polnames[st]][:,1])**2)), float('NaN'))
                        # Flag data
                        if do_loadflag == 'yes':
                            flagfile = str(baseline)+"_"+str(st)+"_"+str(visstart)+"-"+str(visend)+'_'+str(IF+1)+'_flags.npy'
                            flags = np.load(path2folder+flagfile)
                            vis_array[np.where(flags > 0)] = float('NaN')
                            del flags
                            os.remove(path2folder+flagfile)
                        # Save visibilities
                        #print "SAVED: ", vis_output[label]#vis_array[label][1]
                        np.save(vis_output[label], vis_array[label])

    print 'Finished reading FITS file'
    for f in vis_output.values():
         f.close()
    for f in times_output.values():
         f.close()


def visappend3(visnumdict, ntimescans, wizdata, starttimerang, endtimerange, polnames):

	# We already have:
	# 1. a dictionary that contains the baseline and number of timescans associated with it...(ntimescans)
	# 2. a dictionary that contains the baseline and each timescan associated with it that we want to plot (visnumdict)

	# So first I will have to take visnumdict and for each baseline, split the visibility numbers into seperate pages
	# according to timeperage (so long as totalnumberoftimescans is > timeperage)...also do that with ntimescans

	timeblockdict = collections.OrderedDict()
	visnumdict2 = collections.OrderedDict()

	for bline, visnums in visnumdict.items():
		tmp = []
		totntimescans = int(len(visnums))
		if timeperpage >= totntimescans:
			visstart = 0
			vissend = totntimescans
			timeblock = [[visstart, vissend]]
			visnumdict2[bline] = [visnumdict[bline]]
		if timeperpage < totntimescans:
			timeblock = []
			pagetimelist = getpagetimelist(totntimescans, timeperpage)

			for starttime in pagetimelist:
				visstart = starttime
				vissend = starttime+(timeperpage-1)
				if vissend > totntimescans:
					vissend = totntimescans
				if vissend != visstart:
					timeblock.append([visstart, vissend])

				tmp.append(visnumdict[bline][visstart:vissend+1])
			visnumdict2[bline] = tmp
		timeblockdict[bline] = timeblock

	# 1. vis_out[label] = file pointer to pytables visibility file
	# 2. vis_array[label] = pointer to numpy array to be saved within pytables file
	# 3. times_out=[label] = file pointer to pytables time file
	# 4. times_array[label] = pointer to numpy times array to be saved within pytables file
	# 5. countdic[label] = to count each visibility of each page (i.e. baseline/time)
	# 6. label_list = list to keep track of saved file names to return at end of visappend function
	vis_array = collections.OrderedDict()
	vis_out = collections.OrderedDict()
	times_array = collections.OrderedDict()
	times_out = collections.OrderedDict()
	countdict = collections.OrderedDict()
	label_list = []

	timeidentitydic = collections.OrderedDict()
	visidentitydic = collections.OrderedDict()

	timecount = 1
	viscount = 1

	try:
		os.remove(picklepath+'/timesfile')
		os.remove(picklepath+'/visfile')
	except:
		pass

	timesfile = tables.open_file(picklepath+'/timesfile', mode='w')
	visfile = tables.open_file(picklepath+'/visfile', mode='w')

	for bline, timeblock in timeblockdict.items():
		countlist = []
		for tblock in timeblock:
			countlist.append(0)
			labeltime = '{}_{}-{}'.format(bline, tblock[0], tblock[1])

			#timesarrayname = 'timesarray_'+labeltime+'.hdf5'

			# try:
			# 	os.remove(picklepath+'/'+timesarrayname)
			# except:
			# 	pass

			atom = tables.Float64Atom()
			timesarrayidentifier = 'timesarray'+str(timecount)
			timeidentitydic[labeltime] = 'times_file.root.'+str(timesarrayidentifier)
			timecount += 1
			times_array[labeltime] = timesfile.create_earray(timesfile.root, timesarrayidentifier, atom, (0,), labeltime)

			for st in stokes:
				for IF in xrange(IF_start-1,IF_end):
					label = '{}_{}_{}-{}_{}'.format(bline, st, tblock[0], tblock[1], IF+1)

					visarrayname = label+'.hdf5'
					label_list.append(visarrayname)

					# try:
					# 	os.remove(picklepath+'/'+visarrayname)
					# except:
					# 	pass

					visarrayidentifier = 'visarray'+str(viscount)
					visidentitydic[label] = 'visfile.root.'+str(visarrayidentifier)
					viscount += 1
					vis_array[label] = visfile.create_earray(visfile.root, visarrayidentifier, atom, (0,nchan), label)

		countdict[bline] = countlist


	# Create a dictionary to count the timestamp per baseline as we cycle through the visibilities
	indxdict = collections.OrderedDict()
	for bline in visnumdict2.keys():
		indxdict[bline] = 0

	visnumber = 0

	print "\nPopulating visiblities into arrays..."

	for vis in wizdata:
		tmptime = vis.time
		baseline = '{}-{}'.format(vis.baseline[0], vis.baseline[1])

		if tmptime < starttimerang: # this ensures we don't cycle through timeranges we don't want to plot...(time saver)
			visnumber += 1
			continue
		if tmptime > endtimerange:
			break

		if sutab == True:
			if (vis.source in sourceidlist) and (baseline in baselines): # we require the visibility to be one of the sources/baseline we want to plot
				curvisarray = vis.visibility

				if visnumber in visnumdict2[baseline][indxdict[baseline]]: # we require the visibility to be visnumdict - which we created earlier
					tblock = timeblockdict[baseline][indxdict[baseline]]
					labeltime = '{}_{}-{}'.format(baseline, tblock[0], tblock[1])

					time = round(float(vis.time),6)
					times_array[labeltime].append([time])

					for st in stokes:
						for IF in xrange(IF_start-1,IF_end):
							label = '{}_{}_{}-{}_{}'.format(baseline, st, tblock[0], tblock[1], IF+1)

							if amporphas == 'P':
								phas = np.where(curvisarray[IF,:,polnames[st]][:,2] > 0.0, (360./(2*pi))*arctan2(curvisarray[IF,:,polnames[st]][:,1], curvisarray[IF,:,polnames[st]][:,0]), float('NaN'))
								vis_array[label].append([phas])

							if amporphas == 'A':
								amp = np.where(curvisarray[IF,:,polnames[st]][:,2] > 0.0,(np.sqrt((curvisarray[IF,:,polnames[st]][:,0])**2 + (curvisarray[IF,:,polnames[st]][:,1])**2)), float('NaN'))
								vis_array[label].append([amp])

					countdict[baseline][indxdict[baseline]] += 1

					if countdict[baseline][indxdict[baseline]] >= (timeperpage)-1: # this counts the timeperage per baseline
						indxdict[baseline] += 1
					visnumber += 1
				else:
					visnumber += 1
			else:
				visnumber += 1


		if sutab == False:
			if baseline in baselines:
				curvisarray = vis.visibility

				if visnumber in visnumdict2[baseline][indxdict[baseline]]: # we require the visibility to be visnumdict - which we created earlier
					tblock = timeblockdict[baseline][indxdict[baseline]]
					labeltime = '{}_{}-{}'.format(baseline, tblock[0], tblock[1])

					time = round(float(vis.time),6)
					times_array[labeltime].append([time])

					for st in stokes:
						for IF in xrange(IF_start-1,IF_end):
							label = '{}_{}_{}-{}_{}'.format(baseline, st, tblock[0], tblock[1], IF+1)

							if amporphas == 'P':
								phas = np.where(curvisarray[IF,:,polnames[st]][:,2] > 0.0, (360./(2*pi))*arctan2(curvisarray[IF,:,polnames[st]][:,1], curvisarray[IF,:,polnames[st]][:,0]), float('NaN'))
								vis_array[label].append([phas])

							if amporphas == 'A':
								amp = np.where(curvisarray[IF,:,polnames[st]][:,2] > 0.0,(np.sqrt((curvisarray[IF,:,polnames[st]][:,0])**2 + (curvisarray[IF,:,polnames[st]][:,1])**2)), float('NaN'))
								vis_array[label].append([amp])


					countdict[baseline][indxdict[baseline]] += 1

					if countdict[baseline][indxdict[baseline]] >= (timeperpage)-1: # this counts the timeperage per baseline
						indxdict[baseline] += 1
					visnumber += 1
				else:
					visnumber += 1
			else:
				visnumber += 1

	print 'Finished reading FITS file'
	timesfile.close()
	visfile.close()

	del vis_array
	del times_array
	del vis_out
	del times_out

	return label_list, timeidentitydic, visidentitydic


def read_vis(filename):
    # Read multiple arrays stored in .npy files and concatenate them into a single array
    data1 = []
    fp = open(filename, 'rb')

    while 1:
       try:
          data1.append(np.load(fp))
       except:
          break
    return np.asfarray(data1)


########################################################################################
### Makespplot function - a third and final parallelised function that runs per page. It
### first reads in the picklefile of the visibility array and creates a plot of it using
### matlibplot routines.
########################################################################################

def makespplot(s_q,r_q,cpu):

	for job in iter(s_q.get, 'STOP'):

		jobs, timeidentitydic, visidentitydic = job[:]

		for key, value in dict.items(jobs):
			ifpagelist = value
			jobnumber = key

		## Open the visibility file containing all the arrays...
		visfile = tables.open_file(picklepath+'/visfile', mode='r+')
		label = ifpagelist[0][:-5]

		# Call a temparay array to obtain the number of visibilities it contains inorder to create
		# a numpy zero array of the correct dimensions.
		temparray = eval(visidentitydic[label])
		novis = len(temparray)
		temparray = []

		# Plot array will have shape (noIFs,nvis,nchan) - read each IF array into plotarray
		plotarray = np.zeros(shape=(noIFs,novis,nchan))
		for IF in xrange(noIFs):
			#print "Reading in array:", ifpagelist[IF]
			label = ifpagelist[IF][:-5]
			singleifarray = eval(visidentitydic[label])

			# If loading a flag table, load in the appropriate previously created flag array and apply it to
			# the singleifarray
			if do_loadflag == 'yes':
				flagfile = str(label)+'_flags.npy'
				flags = np.load(path2folder+flagfile)
				singleifarray[np.where(flags>0)] = float('NaN')

			plotarray[IF,:,:] = singleifarray[:,:]

		tbsl = ifpagelist[0].split('_')[0]
		tblk = ifpagelist[0].split('_')[2]

		# Now open the times_file and get the appropriate array - this is used to get the yaxis information
		times_file = tables.open_file(picklepath+'/timesfile', mode='r')
		labeltime = tbsl+'_'+tblk
		times_array = eval(timeidentitydic[labeltime])

		tmp = getyticks(wizdata,times_array,sourceidlist, yaxisticks, ticksperpage)
		yticks = tmp[0]
		ylabels = tmp[1]
		yscanlist = tmp[2]

		plfname = ifpagelist[0]

		thisarray = plotarray
		datatype = amporphas
		yticklist = yticks

		times_file.close()
		visfile.close()

		del times_array
		flags = []

		tmpname = plfname.replace(' ','')[:-7].upper()
		print "Creating SPPlot of page: ", tmpname, "on CPU:", (cpu+1)

		fig1 = figure()

		fig1.set_size_inches(11.69,8.27) # Landscape
		fig1.subplots_adjust(wspace=0) # An option close the whitespace between each IF

		if colourscheme == 'jet':
			cmapchoice = cm.jet
		elif colourscheme == 'spectral':
			cmapchoice = cm.spectral
		elif colourscheme == 'gist_rainbow':
			cmapchoice = cm.gist_rainbow
		elif colourscheme == 'cool':
			cmapchoice = cm.cool
		elif colourscheme == 'summer':
			cmapchoice = cm.summer
		elif colourscheme == 'winter':
			cmapchoice = cm.winter
		elif colourscheme == 'anglemap':
			cmapchoice = anglemap
		else:
			cmapchoice = cm.jet

		pltype    = '.pdf'
		DPI       = 500
		fsize     = 8           # fontsize
		stlaboff  = 15
		chnspa = len(thisarray[0][0])/4


		### Create both x-axis (channels) ticks and tick labels
		xtck,xtckl = [],[]
		if nchan >= 8:
			for i in xrange(int(nchan/chnspa)):
				xtckl.append(int((i+1)*chnspa))
				xtck.append(int(((i+1)*chnspa)))

		else:
			for i in xrange(nchan):
				xtckl.append(int(i+1))
				xtck.append(int(i))

		### Set amplitude plotting limits:
		minim = []
		maxim = []
		dmean = []
		rms = []
		missingif = []
		zerominimums = []

		for IFs in xrange(noIFs):
			try:
				mini = min(thisarray[IFs][~np.isnan(thisarray[IFs])])
				if mini != 0.0:
					minim.append(mini)
					zerominimums.append(False)
				else:
					thisarraynon = thisarray[IFs][~np.isnan(thisarray[IFs])]
					mini = min(thisarraynon[np.nonzero(thisarraynon)])
					minim.append(mini)
					zerominimums.append(True)

				maxim.append(max(thisarray[IFs][~np.isnan(thisarray[IFs])]))
				dmean.append(mean(thisarray[IFs][~np.isnan(thisarray[IFs])]))
				rms.append(std(thisarray[IFs][~np.isnan(thisarray[IFs])]))
			except:
				#print "\nNo visibilities exist in this IF"
				missingif.append(IFs)

		# Average over all IFs to include in plot title
		avmin = sum(minim)/len(minim)
		avmax = sum(maxim)/len(maxim)
		avdmean = sum(dmean)/len(dmean)
		avrms = sum(rms)/len(rms)

		for i in xrange(len(missingif)):
			minim.insert(missingif[i],avmin)
			maxim.insert(missingif[i],avmax)
			dmean.insert(missingif[i],avdmean)
			rms.insert(missingif[i],avrms)

		ifwminamp = minim.index(min(minim))+1
		ifwmaxamp = maxim.index(max(maxim))+1

		vvmin = []
		vvmax = []
		vvmin = minim
		vvmax = maxim

		### Section to display the percentage that each IF has been flagged
		flagpercent = []
		nannumlist = []
		for IFs in xrange(noIFs):
			nannum = np.count_nonzero(np.isnan(thisarray[IFs]))
			nannumlist.append(nannum)

		visbychan = (len(thisarray[0])*len(thisarray[0][0]))

		for nannum in nannumlist:
			percentage = nannum*(100./visbychan)
			flagpercent.append(percentage)

		pageflagpercent = sum(flagpercent)*(1/(float(noIFs)))

		minvalueallifs = min(vvmin)
		maxvalueallifs = max(vvmax)

		subplts = []
		count = 1
		combnamedict = {}

		# Create a second figure without any of axis/labels/etc for the combined plot
		if makecombinedplot == 'yes':
			fig_comb = plt.figure(figsize=(5,5))
			fig_comb.subplots_adjust(wspace=0)
			count = 1

			for IFs in xrange(1,noIFs+1):
				comb = fig_comb.add_subplot(1,noIFs,count)

				if scale == 'std' and scale_over_all_IFs == 'yes':
					comb.imshow(thisarray[IFs-1],aspect='auto',interpolation='nearest',origin='lower',cmap=cmapchoice,vmin=minvalueallifs, vmax=maxvalueallifs)

				if scale == 'std' and scale_over_all_IFs == 'no':
					comb.imshow(thisarray[IFs-1],aspect='auto',interpolation='nearest',origin='lower',cmap=cmapchoice,vmin=vvmin[count-1], vmax=vvmax[count-1])

				if scale == 'log' and scale_over_all_IFs == 'yes':
					comb.imshow(thisarray[IFs-1],aspect='auto',interpolation='nearest',origin='lower',cmap=cmapchoice,norm=LogNorm(vmin=0.05*avdmean, vmax=maxvalueallifs))

				if scale == 'log' and scale_over_all_IFs == 'no':
					comb.imshow(thisarray[IFs-1],aspect='auto',interpolation='nearest',origin='lower',cmap=cmapchoice,norm=LogNorm(vmin=0.01*avdmean, vmax=maxim[count-1]))

				if scale == 'linear' and scale_over_all_IFs == 'yes':
					comb.imshow(thisarray[IFs-1],aspect='auto',interpolation='nearest',origin='lower',cmap=cmapchoice,vmin=minvalueallifs, vmax=maxvalueallifs)

				if scale == 'linear' and scale_over_all_IFs == 'no':
					comb.imshow(thisarray[IFs-1],aspect='auto',interpolation='nearest',origin='lower',cmap=cmapchoice,vmin=minim[count-1], vmax=maxim[count-1])


				comb.xaxis.set_ticklabels([], visible=False)
				comb.yaxis.set_ticklabels([], visible=False)
				comb.set_xticks(xtck)
				comb.set_yticks(yticklist)
				comb.plot([0,len(thisarray[0][0])],[yscanlist,yscanlist], color ='black', linewidth=0.8, linestyle="--", alpha = 0.5)

				if count > 1:
					comb.get_yaxis().set_visible(False)
				count += 1


			bsl = plfname[:3]
			stoke = plfname[4:6]

			comb_name = plfname.replace(' ','')[:-6].upper()
			comb_name = 'combined_'+str(comb_name)+'.png'
			combnamedict = {jobnumber:str(comb_name)}
			fig_comb.savefig(path2folder+comb_name, bbox_inches = 'tight')

			comb.cla()
			fig_comb.clf()
			close(fig_comb)

		# This will create the subplots on each page and plot them accordingly
		if onlyplotcombined == 'no':
			count = 1

			for IFs in xrange(1,noIFs+1):
				tmp = fig1.add_subplot(1,noIFs,count)

				if scale == 'std' and scale_over_all_IFs == 'yes':
					cbarimage = tmp.imshow(thisarray[IFs-1],aspect='auto',interpolation='nearest',origin='lower',cmap=cmapchoice,vmin=minvalueallifs, vmax=maxvalueallifs)

				if scale == 'std' and scale_over_all_IFs == 'no':
					cbarimage = tmp.imshow(thisarray[IFs-1],aspect='auto',interpolation='nearest',origin='lower',cmap=cmapchoice,vmin=vvmin[count-1], vmax=vvmax[count-1])

				if scale == 'log' and scale_over_all_IFs == 'yes':
					cbarimage = tmp.imshow(thisarray[IFs-1],aspect='auto',interpolation='nearest',origin='lower',cmap=cmapchoice,norm=LogNorm(vmin=minvalueallifs, vmax=maxvalueallifs))

				if scale == 'log' and scale_over_all_IFs == 'no':
					cbarimage = tmp.imshow(thisarray[IFs-1],aspect='auto',interpolation='nearest',origin='lower',cmap=cmapchoice,norm=LogNorm(vmin=minim[count-1], vmax=maxim[count-1]))

				if scale == 'linear' and scale_over_all_IFs == 'yes':
					cbarimage = tmp.imshow(thisarray[IFs-1],aspect='auto',interpolation='nearest',origin='lower',cmap=cmapchoice,vmin=minvalueallifs, vmax=maxvalueallifs)

				if scale == 'linear' and scale_over_all_IFs == 'no':
					cbarimage = tmp.imshow(thisarray[IFs-1],aspect='auto',interpolation='nearest',origin='lower',cmap=cmapchoice,vmin=minim[count-1], vmax=maxim[count-1])


				xlabelminmax = "Channels \n Min: %.1g \n Max: %.1g \n (Jy)" % (minim[count-1],maxim[count-1])
				tmp.set_xlabel(xlabelminmax, fontsize=fsize)
				tmp.set_ylabel('Observation Time', fontsize=fsize)
				tmp.set_title("IF "+str(selected_IFs[IFs-1])+": "+str('%.2f' % flagpercent[IFs-1])+"% Flags", fontsize=(fsize-1))
				tmp.set_xticks(xtck)
				tmp.set_xticklabels(xtckl,fontsize=fsize*0.75)
				tmp.set_yticks(yticklist)
				tmp.set_yticklabels(ylabels,fontsize=fsize*0.85)
				plot([0,len(thisarray[0][0])],[yticklist,yticklist], color ='black', linewidth=0.5, linestyle="--", alpha = 0.5)
				plot([0,len(thisarray[0][0])],[yscanlist,yscanlist], color ='black', linewidth=0.5, linestyle="-", alpha = 0.5)
				if count > 1:
					tmp.get_yaxis().set_visible(False)
				count += 1

			### Create the colourbar
			cax = fig1.add_axes([0.91,0.1,0.02,0.8]) #Specify colourbar position, width and height
			cbar = fig1.colorbar(cbarimage, cax=cax)
			cbar.ax.tick_params(labelsize=fsize)

			if datatype == 'A':
				datatype = 'AMP'
				units = 'Jy'
			if datatype == 'P':
				datatype = 'PHS'
				units = 'Deg'

			bsl = plfname[:3]
			stoke = plfname[4:6]
			threesigabovemean = float((3*avrms)+avdmean)
			bottomtext = str(datatype)+' BSL:'+str(bsl)+' '+str(stoke)+', Mean: '+str('%.5f'%avdmean)+', RMS: '+str('%.5f'%avrms)+', Max: '+str('%.5f'%maxvalueallifs)+', 3*Sigma+Mean: '+str('%.2f'%threesigabovemean)+' '+str(units)

			if scale == 'std' and scale_over_all_IFs == 'yes':
				righthandtext = 'Linear within 3 std: IFs all to scale'
			if scale == 'std' and scale_over_all_IFs == 'no':
				righthandtext = 'Linear within 3 std: IFs scaled independantly'
			if scale == 'log' and scale_over_all_IFs == 'yes':
				righthandtext = 'Log Scale: all IFs to scale'
			if scale == 'log' and scale_over_all_IFs == 'no':
				righthandtext = 'Log Scale: IFs scaled independantly'
			if scale == 'linear' and scale_over_all_IFs == 'yes':
				righthandtext = 'Linear Scale: all IFs to scale'
			if scale == 'linear' and scale_over_all_IFs == 'no':
				righthandtext = 'Linear Scale: IFs scaled independantly'

			plfname = plfname.replace(' ','')[:-7].upper()
			pdfname = str(outfilename)+'_'+str(plfname)+str(pltype)
			pngname = str(outfilename)+'_'+str(plfname)+'.png'

			fig1.text(0.98,0.5, righthandtext, rotation='vertical', verticalalignment='center',fontsize=fsize)
			fig1.text(0.5, 0.935, bottomtext, fontsize=fsize*1.3,horizontalalignment='center')
			fig1.suptitle(str(outfilename)+'_'+plfname, fontsize=fsize*1.7)
			pp = PdfPages(path2folder+pdfname)
			savefig(pp, format='pdf', dpi=DPI)

			if outputsingle_pngs == 'yes' and pngbbox_inches == 'tight':
				fig1.savefig(path2folder+pngname, bbox_inches='tight')
			if outputsingle_pngs == 'yes' and pngbbox_inches == 'none':
				fig1.savefig(path2folder+pngname)

			pp.close()
			tmp.cla()
			fig1.clf()

			close(fig1)

			plfnamedict = {jobnumber:str(pdfname)}

		del temparray
		del thisarray
		del plotarray
		del singleifarray

		if onlyplotcombined == 'yes':
			plfnamedict = {}

		r_q.put((plfnamedict, zerominimums, pageflagpercent, combnamedict))



#######################################################################################
#######################################################################################
### Main Code #########################################################################

version_number = '1.3.2'
version_date = "14/09/16"

if len(sys.argv) > 1:
    if sys.argv[1] == '--v' or sys.argv[1] == '--version':
        print '\nThis is SPPlot %s dated: %s\n' % (version_number, version_date)
        sys.exit(0)
    else:
        inputfile = sys.argv[1]
else:
    inputfile = "SPPlot_input.py"


print '\n Started running SPPlot version: %s (%s)' % (version_number, version_date), 'on %s' % strftime("%d-%b-%y"), 'at:', strftime("%H:%M:%S", localtime()),'\n'

#######################################################################################
#### Section to retrieve information about the dataset and load inputs from input file!
#######################################################################################

execfile(inputfile)

print "\nYour input file is: ", inputfile
print "Your AIPS USER NUMBER is: ", AIPS.userno

try:
	AIPS.userno
	Name
	Klass
	Disk
	Seq
	path2folder
	picklepath
	IF
	IF_start
	IF_end
	choosesources
	if choosesources == 'choose':
		specifysources
	amporphas
	outfilename
	stokes
	timeperpage
	yaxisticks
	if yaxisticks == 'tickperpage':
		ticksperpage
	timerange
	scale_over_all_IFs
	scale
	colourscheme
	choosebaselines
	if choosebaselines == 'choose':
		baselines
	makecombinedplot
	onlyplotcombined
	outputsingle_pngs
	pngbbox_inches
	do_loadflag
	flag_tables
except NameError:
	print " Please ensure ALL input variables have been specified in the input file"
	print " Aborting!\n"
	sys.exit(0)

if onlyplotcombined == 'yes' and makecombinedplot == 'no':
	print "You have asked to plot nothing - set makecombinedplot = 'yes' or onlyplotcombined = 'no'"
	sys.exit()

if amporphas == 'P' and scale == 'log' or amporphas == 'P' and scale == 'std':
	print 'You cannot plot the phases on a log/std scale!'
	print 'Please change choose scale = linear for phase plots...'
	print 'Aborting...'
	sys.exit(0)


if do_loadflag == 'yes':
	outfilename = outfilename+'_FG'
	if len(flag_tables) == 1:
		outfilename += str(flag_tables[0])
	elif len(flag_tables) > 1:
		outfilename += str(flag_tables[0])
		for i in xrange(1,len(flag_tables)):
			outfilename += '+'+str(flag_tables[i])

uvdatav = [Name, Klass, Disk, Seq]
uvdata = AIPSUVData(uvdatav[0],uvdatav[1],uvdatav[2],uvdatav[3])
uvdata = AIPSUVData(Name, Klass, Disk, Seq)
wizdata = WizAIPSUVData(Name, Klass, Disk, Seq)

try:
	os.mkdir(picklepath)
except OSError:
	os.system('rm '+picklepath+'/*')
	pass

# Figure out number of Stokes and polarizations
npol = len(uvdata.stokes)
print " Number of Polarizations (npol): %i" % npol
polnames = collections.OrderedDict()
for x in xrange(npol):
    polnames[str(uvdata.stokes[x])] = x
# Below is just to print out the stokes in a nice way :)
string = ""
for x in xrange(npol):
    if x != npol-1:
        string += (str(uvdata.stokes[x])+ ", ")
    else:
        string += str(uvdata.stokes[x])
print " Polarizations: %s" % string


# figure out number of IFs
if 'IF' in uvdata.header['ctype']:
    nif = uvdata.header['naxis'][uvdata.header['ctype'].index('IF')]
    print " Number of IFs (nif): %i" % nif
else:
    print " Keyword IF not found in header, assuming number of IFs is 1"
    nif = 1


# figure out number of channels
nchan = uvdata.header['naxis'][uvdata.header['ctype'].index('FREQ')]
print " Number of Channels (nchan): %i" % nchan


# Figure out number of baselines
nant = len(uvdata.antennas)
nbase = ((nant*(nant-1))/2)
nacbase = nbase + nant
print " Number of Possible Baselines (nbase): %i" % nacbase

newsource = []
source_list = {}

# First try and see whether there are any SU tables, and get source numbers and names from that.
try:
	nsource = len(wizdata.sources)
	print " Number of Possible Sources (nsource): %i" % nsource

	for tab in wizdata.tables:
	    if tab[1] == 'AIPS SU' or tab[1] == 'SU':
	    	sutable = wizdata.table('SU',tab[0])
	for row in sutable:
		if row.source.strip() in wizdata.sources:
			newsource.append([row.id__no,row.source.strip()])

	source_list = dict(newsource)

	sourceidlist = []
	nx = wizdata.table('NX',1)

	for i in nx:
		source_id = i['source_id']
		if source_id not in sourceidlist:
			sourceidlist.append(source_id)

	for key in source_list.keys():
		if key not in sourceidlist:
			del source_list[key]

	sutab = True
	nsource = len(source_list)
	print " Number of Actual Sources: ",nsource
	print "\nNames of sources within the actual dataset/catalogue entry:"
	for s in source_list:
		if not s == len(source_list):
			print " %i: %s" % (s,source_list[s])
		else:
			print " %i: %s" % (s, source_list[s])


except:
    print "\n No SU table found... "
    print " Assuming single source...getting source name from header"
    sutab = False
    header = wizdata.header
    source_list[1] = header['object']
    nsource = 1
    print " %i: %s" % (1, source_list[1])


if choosesources == 'choose':
	nsource = len(specifysources)
	for key, value in source_list.items():
		if value not in specifysources:
			del source_list[key]

print "\nThe following sources will be plotted: (if they fall within you're chosen timerange...)"
for s in source_list:
	print " %i: %s" % (s,source_list[s])

sourceidlist = source_list.keys()


# If timerange has been set then convert the hms format into decimal format
if timerange:
	print '\nYou have selected to plot between times: %s and %s' % (timerange[0], timerange[1])
	starttimerange = hms2dectime(timerange[0])
	endtimerange = hms2dectime(timerange[1])
if not timerange:
	starttimerange = 0.
	endtimerange = 253.

if choosebaselines == 'all':
	# List of baselines
    antennas = []
    antab = uvdata.table('AN',1)
    for row in antab:
        antennas.append(row['nosta'])
    baselines = []
    for i in range(len(antennas)):
        for j in range(i+1, len(antennas)):
            baselines.append('{0}-{1}'.format(antennas[i], antennas[j]))
    for bline in baselines:
        if bline == '1-2':
            del baselines[baselines.index(bline)]

    print "\n ALL %s baselines have been selected for spplotting..." % len(baselines)

    ntimescans = {}
    ntime = 0
    visnumber = 0
    visnumdict = {}
    vistimedic = {}
    for bline in baselines:
        ntimescans[bline] = 0
        visnumdict[bline] = []
        vistimedic[bline] = []
    for vis in wizdata:
        tmptime = vis.time
        if tmptime < starttimerange:
        	visnumber += 1
        	continue
        if tmptime > endtimerange:
        	break
    	try:
	    	if vis.source not in sourceidlist:
    			visnumber += 1
    			continue
    	except KeyError:
    		#visnumber += 1
    		pass

        if starttimerange <= tmptime and endtimerange >= tmptime:
	        bline = "%i-%i" % (vis.baseline[0], vis.baseline[1])

	        if vis.baseline[0] == vis.baseline[1]:
	        	visnumber += 1
	        	continue
	        if bline == '1-2':
	        	visnumber += 1
	        	continue

	        visnumdict[bline].append(visnumber)
	        vistimedic[bline].append(tmptime)
	        ntimescans[bline] += 1
	        ntime += 1
	        visnumber += 1
        else:
	    	visnumber += 1
    ntimescans = collections.OrderedDict(sorted(ntimescans.items()))
    print "\nList of ALL baselines with corresponding number of time scans:", ntimescans


# Testing for specified baselines.
if choosebaselines == 'choose':
    print "\n SPECIFIC baselines selected for spplotting..."

    ntimescans = {}
    ntime = 0
    visnumber = 0
    visnumdict = {}
    vistimedic = {}
    for bline in baselines:
        ntimescans[bline] = 0
        visnumdict[bline] = []
        vistimedic[bline] = []
    for vis in wizdata:
        tmptime = vis.time
    	if tmptime < starttimerange:
        	visnumber += 1
        	continue
        if tmptime > endtimerange:
        	break
    	try:
	    	if vis.source not in sourceidlist:
	    		visnumber += 1
    			continue
    	except KeyError:
    		pass

        if starttimerange <= tmptime and endtimerange >= tmptime:
        	bline = "%i-%i" % (vis.baseline[0], vis.baseline[1])

	        if bline in baselines:
	        	visnumdict[bline].append(visnumber)
	        	vistimedic[bline].append(tmptime)
	        	ntimescans[bline] += 1
	        	ntime += 1
	        	visnumber += 1
	        else:
	        	visnumber += 1
        else:
	    	visnumber += 1
    ntimescans = collections.OrderedDict(sorted(ntimescans.items()))
    print "\nList of CHOSEN baselines with corresponding number of time scans:"
    for baseline, ntimes in ntimescans.items():
    	print baseline, " - ", ntimes

# Added to ensure baseline with NO visibilities are included...
for bline, numofvis in ntimescans.items():
	if numofvis == 0:
		print "\nNo visiblities exist for baseline %s" % bline
		del ntimescans[bline]
		del visnumdict[bline]
		del vistimedic[bline]

visnumdict = collections.OrderedDict(sorted(visnumdict.items()))

nostokes = len(stokes)
nbline = len(baselines)
noIFs = (IF_end-IF_start)+1
selected_IFs = range(IF_start, IF_end+1)
orderedpols = get_ordered_pols(uvdata)

nvis = 0
pagedic = {}
for bline, ntime in ntimescans.items():
	nvis += ntime
	if ntime % timeperpage == 0:
		pagedic[bline] = ntime/timeperpage
	else:
		pagedic[bline] = (ntime/timeperpage)+1

numoffiles = ((sum(pagedic.values())*noIFs)*nostokes)+sum(pagedic.values())

if ntime == 0:
	print '\nNo data in your selected timerange for your chosen baselines/sources...'
	print 'Aborting...'
	sys.exit(0)

try:
    mem_stats = computer_memory(path2folder)
except:
    print "\n Sorry, computer_memory() definition does not work on your system!"

if 'mem_total' in mem_stats:
    print "\nSystem Memory Information:"
    print "Total Memory  :    %s" % array_size(mem_stats['mem_total']*1000)
    print "Used Memory   :    %s" % array_size(mem_stats['mem_used']*1000)
    print "Free Memory   :    %s" % array_size(mem_stats['mem_free']*1000)
    print "Total Swap    :    %s" % array_size(mem_stats['swap_total']*1000)
    print "Used Swap     :    %s" % array_size(mem_stats['swap_free']*1000)
    print "Free Swap     :    %s" % array_size(mem_stats['swap_used']*1000)


# Section to estimate the size of the arrays and hence the memory usage
if do_loadflag == 'no':
	if timeperpage >= max(ntimescans.values()):
		pred_array = timeperpage*nchan*8*nif
	else:
		pred_array = max(ntimescans.values())*nchan*8*nif
if do_loadflag == 'yes':
	if timeperpage >= max(ntimescans.values()):
		pred_array = (timeperpage*nchan*8*nif)+(nchan*timeperpage)
	else:
		pred_array = (max(ntimescans.values())*nchan*8*nif)+(nchan*max(ntimescans.values()))

print "\nPredicted memory usage is approximately: %s" % array_size(int(pred_array))

########################################################################################################
### Section 0: Load in flag tables if selected to do so with the do_loadflag option. This section is
### parallelised over baseline and timeperage - it will cycle through entire flag table, create a numpy
### array of data (putting 1 where their is a flag), save the numpy files per bline,timeperpage,st,if.
########################################################################################################

if do_loadflag == 'yes':

    ### First check flag tables exist
    notabs = []
    for fg in flag_tables:
        try:
            fgtab = wizdata.table('FG',fg)
        except IOError:
            print '\nERROR: Flag table %i selected for loading, but it does not seem to exist' % fg
            print "ERROR: You should either remove this from the flag_tables list or set do_loadflag = 'no'"
            print "ERROR: Removing table %i from flag table list" % fg
            notabs.append(fg)

    for tab in notabs:
        flag_tables.remove(tab)

    if not flag_tables:
        print '\nERROR: No flag tables selected or no flag tables found.'
        print 'ERROR: Aborting loading previous flag tables and continuing with flagging. \n'
        flagsexist = False

    else:
        flagsexist = True
        joblist0 = []
        jobnum = 1
        npol = len(stokes)

        print "\n You have chosen to load in a flag file...specifically table number(s):", flag_tables

        for key in orderedpols.keys():
        	if key not in stokes:
        		del orderedpols[key]
        		del polnames[key]

        tmp = np.array(np.sort(orderedpols.values()))

        counter = 0
        indxorderedpol = {}
        for key, value in polnames.items():
        	if value > counter:
        		indxorderedpol[str(key)] = counter
        	else:
        		indxorderedpol[key] = value
        	counter += 1

        # Create a joblist --- number of jobs = number of baseline times number of pages
        for baseline, timearray in vistimedic.items():
        	totntimescans = len(timearray)

        	if timeperpage >= totntimescans:
        		totaljobs = len(vistimedic)
        		visstart = 0
        		visend = totntimescans

        		times_array = np.array(timearray)
        		joblist0.append([Name, Klass, Disk, Seq, times_array, flag_tables, sourceidlist, baseline, orderedpols, indxorderedpol, stokes, nchan, noIFs, npol, visstart, visend, path2folder, jobnum, totaljobs])
        		jobnum += 1

        	if timeperpage < totntimescans:
        		pagetimelist = getpagetimelist(totntimescans,timeperpage)
        		totaljobs = len(vistimedic)*len(pagetimelist)

        		for starttime in pagetimelist:
	        		visstart = starttime
	        		visend = starttime+(timeperpage-1)
	        		if visend > totntimescans:
	        			visend = totntimescans

	        		times_array = np.array(timearray[visstart:visend])
        			joblist0.append([Name, Klass, Disk, Seq, times_array, flag_tables, sourceidlist, baseline, orderedpols, indxorderedpol, stokes, nchan, noIFs, npol, visstart, visend, path2folder, jobnum, totaljobs])
        			jobnum += 1

        send_q = mp.Queue()
        recv_q = mp.Queue()
        ncpus = mp.cpu_count()

        for i in xrange(len(joblist0)):
        	#print "Sending Job #%.i of %.i to Queue" % ((i+1), len(joblist0))
        	send_q.put(joblist0[i])

        for cpu in xrange(ncpus):
        	proc = mp.Process(target=load_in_flags, args=(send_q,recv_q,cpu))
        	proc.start()
        	#print 'Starting process on CPU: %.i' % (cpu+1)

        numpyfilelist = []
        for i in xrange(len(joblist0)):
        	numpyfilelist.append(recv_q.get())

        for i in xrange(ncpus):
        	send_q.put('STOP')

        print "\nTime taken to load in flags and save the arrays (hh:mm:ss):", time2hms(time.time()-ti)
        print 'Finished on %s' % strftime("%d-%b-%y"), 'at:', strftime("%H:%M:%S", localtime()),'\n'


###################################################################################################
### OLD Section 1: Actual bit of working paralellised code from here to read in visibilities using
### visappend function - this section is now hashed out...
###################################################################################################

# joblist = []
# jobnumber = 1

# for baseline, visnumarray in visnumdict.items():
# 	totntimescans = len(visnumarray)
# 	for st in stokes:
# 		if timeperpage >= totntimescans:
# 			numofjobs = noIFs*nostokes*nbline

# 			visstart = 0
# 			visend = totntimescans

# 			for IF in xrange(IF_start-1,IF_end):
# 				joblist.append([Name,Klass,Disk,Seq,visstart,visend,baseline,IF,polnames,st,nchan,sourceidlist,jobnumber,visnumarray])
# 				jobnumber += 1

# 		if timeperpage < totntimescans:
# 			pagetimelist = getpagetimelist(totntimescans,timeperpage)
# 			numofjobs = len(pagetimelist)*noIFs*nostokes*nbline

# 			for starttime in pagetimelist:
# 				visstart = starttime
# 				visend = starttime+(timeperpage-1)
# 				if visend > totntimescans:
# 					visend = totntimescans

# 				visnumarray2 = visnumarray[visstart:visend]

# 				for IF in xrange(IF_start-1,IF_end):
# 					joblist.append([Name,Klass,Disk,Seq,visstart,visend,baseline,IF,polnames,st,nchan,sourceidlist,jobnumber,visnumarray2])
# 					jobnumber += 1


# send_q = mp.Queue()
# recv_q = mp.Queue()
# ncpus = mp.cpu_count()

# for i in xrange(len(joblist)):
# 	#print "Sending Job #%.i of %.i to Queue" % ((i+1), len(joblist))
# 	send_q.put(joblist[i])

# for cpu in xrange(ncpus):
# 	proc = mp.Process(target=visappend, args=(send_q,recv_q,cpu))
# 	proc.start()
# 	#print 'Starting process on CPU: %.i' % (cpu+1)

# resultsdict = []
# for i in xrange(len(joblist)):
# 	resultsdict.append(recv_q.get())

# for i in xrange(ncpus):
# 	send_q.put('STOP')

# resultsdict.sort()

# results = []
# for i in xrange(1,len(resultsdict)+1):
# 	results.append((resultsdict[(i-1)])[i])



####################################################################################################
### NEW Section 1: Runs the visappend3 funtion to append to visibilities straight to a pytable style
### file.
####################################################################################################


results, timeidentitydic, visidentitydic = visappend3(visnumdict, ntimescans, wizdata, starttimerange, endtimerange, polnames)

print "\nTime taken to append and save visibilities into arrays (hh:mm:ss):", time2hms(time.time()-ti)
print 'Finished on %s' % strftime("%d-%b-%y"), 'at:', strftime("%H:%M:%S", localtime()),'\n'


#########################################################################################################
### Section 2: A 2nd paralellised section to read in picklefiles and combine IFs into one array which is
### then spploted out onto one page...
#########################################################################################################

import re

numofpages = len(results)/noIFs
pageslist = []

for page in xrange(numofpages):
	start = page*noIFs
	end = start+(noIFs)
	pageslist.append(results[start:end])

listtomerge = []
numofjobs = len(pageslist)
joblist = []

pagesdictionary = []
for i in xrange(1,len(pageslist)+1):
	pagesdic = {i:pageslist[i-1]}
	pagesdictionary.append(pagesdic)

joblist = []
for i in xrange(len(pagesdictionary)):
	job = [pagesdictionary[i],timeidentitydic, visidentitydic]
	joblist.append(job)

send_q = mp.Queue()
recv_q = mp.Queue()
ncpus = mp.cpu_count()

for i in xrange(len(pagesdictionary)):
	print "Sending Job #%.i of %.i to Queue" % ((i+1), len(pagesdictionary))
	send_q.put(joblist[i])

for cpu in xrange(ncpus):
	proc = mp.Process(target=makespplot, args=(send_q,recv_q,cpu))
	proc.start()

resultsdict = []
zeromins = []
percentflagged = []
combsdict = []

for i in xrange(len(pageslist)):
	a = recv_q.get()
	resultsdict.append(a[0])
	zeromins.append(a[1])
	percentflagged.append(a[2])
	combsdict.append(a[3])

for i in xrange(ncpus):
	send_q.put('STOP')

resultsdict.sort()
combsdict.sort()

if onlyplotcombined == 'no':
	results = []
	for i in xrange(1,len(resultsdict)+1):
		results.append((resultsdict[(i-1)])[i])

if makecombinedplot == 'yes':
	combimagelist = []
	for i in xrange(1,len(combsdict)+1):
		combimagelist.append((combsdict[(i-1)])[i])


print "\nTime taken thus far (hh:mm:ss):", time2hms(time.time()-ti)
print 'Finished on %s' % strftime("%d-%b-%y"), 'at:', strftime("%H:%M:%S", localtime()),'\n'

######################################################################################################
### Section 3: Tidy up the results, merge the pages, give statistics, make combined plot if so desired
######################################################################################################

### Make a combined plot page...
if makecombinedplot == 'yes':

	stokesincombinedpages = []
	for st in stokes:
		if st == 'RR' or st == 'LL':
			if ['RR','LL'] in stokesincombinedpages:
				continue
			else:
				stokesincombinedpages.append(['RR','LL'])
		if st == 'RL' or st == 'LR':
			if ['RL','LR'] in stokesincombinedpages:
				continue
			else:
				stokesincombinedpages.append(['RL','LR'])

	pagedictionary, timeranglist = sortcomb_pages(ntimescans,combimagelist)

	for page in pagedictionary:
		imagelist = pagedictionary[page]

		for z in stokesincombinedpages:

			r = z[0]
			l = z[1]
			if r == 'RR' or r == 'LL':
				stk = 'para'
			if r == 'RL' or r == 'LR':
				stk = 'cross'

			antennas = np.unique(np.asarray([baselines_i.split('-') for baselines_i in baselines]).flatten())
			fig2 = plt.figure(figsize=(32,32))
			fig2.subplots_adjust(wspace=0.00, hspace=0.00)

			numofant = len(antennas)

			for i in range(0, (len(antennas)-1)):
			    for j in range(i+1, len(antennas)):

			        bslname="%s-%s" % (antennas[i], antennas[j])

			        plotnum_RR = (i+1)+((j+1)-1)*nant
			        plotnum_LL = (j+1)+((i+1)-1)*nant

			        temp_RR = fig2.add_subplot(nant, nant, plotnum_RR)
			        temp_LL = fig2.add_subplot(nant, nant, plotnum_LL)

			        temp_RR.set_xticks([])
			        temp_LL.set_xticks([])
			        temp_RR.set_yticks([])
			        temp_LL.set_yticks([])

			        temp_RR.set_frame_on(False)
			        temp_LL.set_frame_on(False)

			        if i==0:
			            temp_RR.set_ylabel('%s Ant %s' % (r, antennas[j]), rotation=90, fontsize=20)
			        if j==numofant-1:
			            temp_RR.set_xlabel('%s Ant %s' % (r, antennas[i]), rotation=0, fontsize=20)

			        temp_LL.yaxis.set_label_position("right")
			        temp_LL.xaxis.set_label_position("top")

			        if i==0:
			            temp_LL.set_xlabel('%s Ant %s' % (l, antennas[j]), rotation=0, fontsize=20)
			        if j==numofant-1:
			            temp_LL.set_ylabel('%s Ant %s' % (l, antennas[i]), rotation=90, fontsize=20)


			        try:
		        		tmpRR = str("".join(image for image in imagelist if bslname in image and r in image))
		        		nameRR = path2folder+tmpRR
		        		temp_RR.imshow(read_png(nameRR)[::-1], origin='lower')
		        		if os.path.isfile(nameRR) == True:
		        			os.remove(nameRR)
			        except:
			        	continue

			        try:
		        		tmpLL = str("".join(image for image in imagelist if bslname in image and l in image))
		        		nameLL = path2folder+tmpLL
		        		temp_LL.imshow(read_png(nameLL)[::-1], origin='lower')
		        		if os.path.isfile(nameLL) == True:
		        			os.remove(nameLL)
			        except:
			        	continue


                        combinedpagename1 = outfilename+'_'+timeranglist[page-1]+'_'+str(stk)+'_combined.pdf'
                        combinedpagename2 = outfilename+'_'+timeranglist[page-1]+'_'+str(stk)+'_combined.png'
                        fig2.savefig(path2folder+combinedpagename1, bbox_inches='tight')
                        fig2.savefig(path2folder+combinedpagename2, bbox_inches='tight')
                        print 'Combined plot saved to: ', path2folder+combinedpagename1
                        print 'Combined plot saved to: ', path2folder+combinedpagename2


if onlyplotcombined == 'no':
	### Merge pages into one .pdf file...
	print "\nReady to Merge pages..."

	results = results

	mergepdfs(results)

try:
    os.system('rm -r '+picklepath)
except:
    pass

print "\nTime taken to complete SPPlot (hh:mm:ss):", time2hms(time.time()-ti)
print 'Finished on %s' % strftime("%d-%b-%y"), 'at:', strftime("%H:%M:%S", localtime()),'\n'

percentdataleft = (100-(sum(percentflagged)*(1/float(len(percentflagged)))))

print "The total amount of data left is...%.1f percent" % percentdataleft

zeromins = np.array(zeromins)

if sum(zeromins) > 0:
	print "\n*******************************************"
	print 'WARNING: There are zero-valued amplitudes with positive weights present in this data.\nThese have been ignored by SPPlot.'
	print "*******************************************\n"

count=0
for page in zeromins:
    if True in page:
        print '\nYour zero-valued amplitudes are in page', resultsdict[count].values(), 'and IFs: ',
        for i in xrange(len(page)):
            if page[i] == True:
                print i+1,
        print '\n'
    count += 1
