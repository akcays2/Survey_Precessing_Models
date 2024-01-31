ALL:

There is a variable in all mm.py files called "file_name". This corresponds to the 
output location of the results from the run and should be changed.

NRSur7dq4.h5 needs to be placed either in this directory or in $LAL_DATA_PATH


SXS:
The functions in utils.py that calculate the sxs mismatches have a variable called
"fgw" which is a string containing the location of the sxs h5 file. You will have
to change ALL fgw instances in utils to the your location(s).


NOTE:
We advise setting maxFun=2 to test the code. We also advise using XPHM for testing as it is the fastest.