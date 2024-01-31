#!/bin/bash
approx2Array=("SEOB" "TEOB" "TPHM" "XPHM")
incArray=("0","90")
massArray=("M37.5", "M150")
echo "mass_flag is $1"
echo "The mass is ${massArray[$1]}"
echo "mr_flag is $2"
echo "number of procs is $3"
echo "Approx 1 is NRSur"
echo "Approx 2 is ${approx2Array[$4]}"
echo "Inclination is ${incArray[$5]}"
export OMP_NUM_THREADS=1
NUMBA_NUM_THREADS=1 python mm_sxs_short.py $1 $2 $3 $4 $5