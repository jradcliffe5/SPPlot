# SPPlot!

A data visualisation programme intended for use on radio interferometric data within the field of Astronomy!

Originally desinged for use on data taken with e-MERLIN (enhanced Multi-Element Radio Linked Interferometer Network), the programme is written in ParselTongue - a python based scripting language for formulation within AIPS (Astronomical Image Processing System), however the programme is suitable for use with any radio interferometric array data.

This programme will plot your radio visibility amplitudes or phases in a 3-D plot over both time and frequency space. Its original intention was to mimick the AIPS task 'SPFLG' in viewing your radio data, but to give the user far more control in the resulting plot. This flexibility and its speed when handling large data volumes means SPPlot is a handy tool to view your data at any stage throughout its reduction.

Files include:

1. SPPlot.py - includes all the main functions and processes to execute the programme.
2. SPPlot_input.py - is the input file in which the user chooses their desired options and includes more detailed                                     intructions for use.

To run, (parsetongue must be installed...) in the command line type: % parseltongue SPPlot.py

Most recent version: SPPlot 1.3 dated: 16/05/16 
