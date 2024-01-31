#!/bin/bash

#Call with mm.sh 0/1 0/1 n 0/1/2/3
#where the 0/1 in first position corresponds to light/heavy
#0/1 in the second corresponds to NoMR/Mr
#n is the number of procs
#last arg picks the approx
approx2Array=("SEOB" "TEOB" "TPHM" "XPHM")
incArray=("0","90")
echo "mass_flag is $1"
echo "mr_flag is $2"
echo "number of procs is $3"
echo "Approx 1 is NRSur"
echo "Approx 2 is ${approx2Array[$4]}"
echo "Inclination is ${incArray[$5]}"
export OMP_NUM_THREADS=1
NUMBA_NUM_THREADS=1 python mm_discrete_rand.py $1 $2 $3 $4 $5