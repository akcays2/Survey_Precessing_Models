import numpy as np
import h5py
import matplotlib.pyplot as plt
import json
import math
import os
import sys
# import sxs
import statistics as stat
import random as rand
import lalsimulation as lalsim
import lal
from math import factorial as fact
from scipy import interpolate
import utils as utils
try:
    import EOBRun_module
except:
    print("No TEOB in this env")
import argparse

from joblib import Parallel, delayed
from pycbc.filter import match, overlap
from pycbc.types.frequencyseries import FrequencySeries
from pycbc.types.timeseries import TimeSeries
from pycbc.psd import (
    aLIGODesignSensitivityP1200087,
    AdVDesignSensitivityP1200087,
    EinsteinTelescopeP1600143,
    aLIGOZeroDetHighPower,
)

if __name__ == "__main__":
    mass_flag = int(sys.argv[1])
    mflag = mass_flag
    mass_str = ["M37.5", "M150", "rand"][mass_flag]
    mr_flag = int(sys.argv[2])
    mr_str = ["NoMR", "MR"][mr_flag]
    num_procs = int(sys.argv[3])
    approx_2_flag = int(sys.argv[4])
    approx_2_str = ["SEOB", "TEOB", "TPHM", "XPHM","NR"][approx_2_flag]
    inc_flag=int(sys.argv[5])
    incl = [0,90][inc_flag]
    dist = 500
    maxfun = 1000 #1000
    output_ON = 1
    print(f"Getting data from {mass_str}_params.dat")
    if mass_flag != 2:
        data_file = np.genfromtxt(f"../data_files/discrete_random/{mass_str}_params.dat", names=True)
    else:
        data_file = np.genfromtxt(f"../data_files/discrete_random/random_params.dat", names=True)
    file_name = f"nr_nr_results/{mass_str}_NRSur_{approx_2_str}_{mr_str}_inc{incl}.txt"
    inc = (np.pi*incl/180)
    print(inc)
    print(f"Data will be saved in {file_name}")

    if approx_2_str == "SEOB":
        result = Parallel(n_jobs=num_procs)(delayed(utils.nr_seobV5_mm)(k, data_file, inc, mr_flag, dist, maxfun, mflag)
                                            for k in range(1, len(data_file)+1))
    if approx_2_str == "TEOB":
        result = Parallel(n_jobs=num_procs)(delayed(utils.nr_teob_mm)(k, data_file, inc, mr_flag, dist, maxfun, mflag)
                                            for k in range(1, len(data_file)+1))
    if approx_2_str == "TPHM":
        result = Parallel(n_jobs=num_procs)(delayed(utils.nr_tphm_mm)(k, data_file, inc, mr_flag, dist, maxfun, mflag)
                                            for k in range(1, len(data_file)+1))
    if approx_2_str == "XPHM":
        result = Parallel(n_jobs=num_procs)(delayed(utils.nr_xphm_mm)(k, data_file, inc, mr_flag, dist, maxfun, mflag)
                                            for k in range(1, len(data_file)+1))
    if approx_2_str == "NR":
        result = Parallel(n_jobs=num_procs)(delayed(utils.nr_nr_mm)(k, data_file, inc, mr_flag, dist, maxfun, mflag)
                                            for k in range(1, len(data_file)+1))
                                            

    output = np.array([["Case_No", "mismatch[standard]", "Min[mismatch[fancy]]",
                      "Mean[mismatch[fancy]]", "Max[mismatch[fancy]]", "Sigma[mm(fancy)]"]])

for i in range(0, len(result)):
    res = np.array([[result[i][0], result[i][1], result[i][2],
                   result[i][3], result[i][4], result[i][5]]])
    output = np.append(output, res, axis=0)

    if output_ON:
        np.savetxt(file_name, output, delimiter='\t', fmt="%s")
