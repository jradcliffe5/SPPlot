######################################
### SPPlot input file ################ 
######################### Version: 1.3
######################################
###################### Dated: 16/05/16
######################################
################# Author: Jack Morford
######################################

### SPPlot follows a 'simple' routine. It read in your visibilities, stores them in 
# numpy arrays, and plots the data on 3D plot of frequency x time x amplitude/phase.

# FIRST - if chosen, it will read in one or more flag tables and dump these out as
# numpy arrays (load_in_flags function originally written by Danielle Fenech).
# SECOND - it will append the visibilities into arrays (per IF, per baseline, per stokes,
# per timeperpage) and match the flag array to the visibility array if one is present
# before saving (with numpy.save) this array into the picklepath directory...
# THIRD - it will read in the numpy arrays in parrallel, collating each IF together
# into an array fit for one spplot page, and physically spplot each page using matlibplot
# routines within the makespplot function.
# FOURTH - it merges all the pages into one .pdf file and/or creates a single 'combined'
# plot that includes each baseline on one page. There is one combined plot for every
# timeperage/parallel and cross hands

### This program can be run on both single and multi source files and you may select
# the sources you wish to plot (it will plot them all in time order).  

AIPS.userno = 8020           	            # The AIPS user number the data is on.
Name = 'ALLCALS'             	            # The uvdata name of the catalogue.
Klass = 'UVDATA'                          # The uvdata klass of the catalogue.
Disk = 1                                  # The uvdata disk number of the catalogue (integer).
Seq = 1


path2folder = '/import/cobrasarchive1/red/jmorford/spplottesting/spplots/'
                                          # Choose output directory for your plots to be stored. 

picklepath = str(path2folder)+'picklefiles'
                                          # Choose output directory for your pickled dictionaries to be stored.

choosesources = 'all'		                  # Please select either 'all' or 'choose'

specifysources = ['J2007+404']	          # Must be strings in python list format.

choosebaselines = 'choose'	                  # Please select either 'all' or 'choose'

baselines = ['6-8','6-9','7-8','8-9']     # Specify your baselines in the format ['1-2','5-6','7-8',etc]
                                          # NB: if 5 baselines or fewer are chosen the program will UVCOP 
                                          # the dataset for your chosen baselines - this is intended to save 
                                          # time when appending the visibilities into the memory. If more than 
                                          # 5 baselines are selected, the entire UV datafile including all 
                                          # baselines will be read into the memory and pickled into a pickle file.


outfilename = 'phasecal'      	    # This is the base name of your output plots.


stokes = ['RR', 'LL']		                  # Must be strings seperated by commas!


timerange = ['1 23 40 0', '2 1 00 00']
#timerange = []                           # Please select the timerange you wish to plot - e.g. a list of two strings
                                          #  1st being start time, 2nd being the end time
                                          # e.g. ['0 4 30 5', '0 8 54 57'] i.e. each string = 'day hour min sec'
                                          # To plot all times, leave the timerange list empty!

# Plotting Options:

scale = 'linear'	         	              # Choose whether to scale the amplitudes linearly ('linear'), 
                                          # logarithmically ('log') or min/max within 3 std of mean ('std') - 
                                          # a log scale may show greater detail if the differences in min and max 
                                          # amplitudes are very large. If 'std' is chosen, the max and min amplitudes 
                                          # will be taken to be the mean +/- three stand deviations.

scale_over_all_IFs = 'yes'	                # Choosing to scale over all IFs gives a true representation of your data 
                                          # but since RFI can vary severly from IF to IF it is likely that your plots 
                                          # will show less detail for those IFs less affected by RFI. If 'no' the 
                                          # program will scale each IF individually according to their own maximum 
                                          # and minimum amplitudes - this will give a good indication of RFI within 
                                          # individual IFs but note that the colorbar scale will correspond only to 
                                          # the last IF. Underneath each IF there is a min and max amplitude value 
                                          # for each IF - this will give you an idea of the differences between each 
                                          # IF if 'no' is selected. 
colourscheme = 'jet'                      # This allows you to choose the colour scheme adopted for the colour 
                                          # scaling of your plots. Currently supported options are jet, spectral, 
                                          # gist_rainbow, cool, summer or winter. To see what these colour schemes
                                          # look like see http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
                                          # This will default to the 'jet' colourscheme if none is specified.


amporphas = 'A'			                      # Specify your plots to display either amplitude: 'A' or phase: 'P'. 
                                          # PLEASE DONT SET 'P' WITH SCALE = 'LOG'! It will fail...

timeperpage = 2000		                    # Set the amount of visibilities to use per page - i.e. the amount of time 
                                          # (y-axis) to squeeze onto a sheet of A4. Something between 1000-2000 is reasonable.

IF = 8				                            # How many IFs are present in your dataset i.e. for e-Merlin L-Band = 8, C-Band = 16.

IF_start = 1 			                        # Choose starting IF to plot.
IF_end = 8			                          # Choose last IF to plot.


makecombinedplot = 'yes'                  # If 'yes' SPPlot will make thumbnail like .png files of each page and plot them all 
                                          # onto one single page!

onlyplotcombined = 'no'                   # If 'yes' SPPlot will bypass the rest of the makespecplot function to only create
                                          # the combined plot this will speed things up!
                                          # NB: if 'no' and makecombined plot = 'no' - nothing will be plotted.

do_loadflag = 'yes'                        # Choose to apply previous flag tables to the data before flagging
                                          # Options are 'yes' (load tables specified in flag_tables) or 
                                          # 'no' (don't load any flag tables)

flag_tables = [2]                         # Give a comma-separated list of flag tables to load if you wish to load
                                          # more than one at once... e.g. [1,2]
