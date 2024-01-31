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
#import EOBRun_module
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


# freqs=freq_data["f0(Hz)_for_1Msun"]
# inc = 0
# mr_flag = 1
# distance = 500
# maxFun = 1000
# mflag = 0
# result = Parallel(n_jobs=1)(
#     delayed(utils.sxs_xphm_mm)(i, data, inc, mr_flag, distance, maxFun,mflag,freqs) for i in range(list_length))

if __name__ == "__main__":
    sxs_list_path = "../data_files/sxs/SXS_large_chip_Res_List.txt"
    freq_list = "freqs.txt"
    with open(sxs_list_path, "r") as f:
        # format: SXS_sim_ID	chi_p0	chi_pMax	Max_Res
        data = np.loadtxt(f, skiprows=1)
        simNo = data[:, 0]
        simResLev = data[:, 3]

    list_length = len(simNo)
    mass_flag = int(sys.argv[1])
    mflag = mass_flag
    mass_str = ["M37.5", "M150"][mass_flag]
    mr_flag = int(sys.argv[2])
    fdata=np.genfromtxt(f"../data_files/sxs/{mass_str}_f_peak.txt",names=True)
    f_peaks=fdata["f_peak"]
    mr_str = ["NoMR", "MR"][mr_flag]
    num_procs = int(sys.argv[3])
    approx_2_flag = int(sys.argv[4])
    incl=[0,90][int(sys.argv[5])]
    approx_2_str = ["SEOB", "TEOB", "TPHM", "XPHM"][approx_2_flag]
    #incl = 90
    dist = 500
    maxfun = 1000
    output_ON = 1
    file_name = f"results_long/{mass_str}_SXS_{approx_2_str}_{mr_str}_{maxfun}_inc{incl}.txt"
    inc = (np.pi*incl/180)
    print(inc)
    print(f"Data will be saved in {file_name}")

    if approx_2_str == "SEOB":
        result = Parallel(n_jobs= num_procs)(
            delayed(utils.sxs_seobV5_mm)(i, data, inc, mr_flag, dist, maxfun, mflag,f_peaks) for i in range(list_length))
    if approx_2_str == "TEOB":
         result = Parallel(n_jobs= num_procs)(
            delayed(utils.sxs_teob_mm)(i, data, inc, mr_flag, dist, maxfun, mflag,f_peaks) for i in range(list_length))
    if approx_2_str == "TPHM":
         result = Parallel(n_jobs= num_procs)(
            delayed(utils.sxs_tphm_mm)(i, data, inc, mr_flag, dist, maxfun, mflag,f_peaks) for i in range(list_length))
    if approx_2_str == "XPHM":
         result = Parallel(n_jobs= num_procs)(
            delayed(utils.sxs_xphm_mm)(i, data, inc, mr_flag, dist, maxfun, mflag,f_peaks) for i in range(list_length))

    output = np.array([["Case_No", "mismatch[standard]", "Min[mismatch[fancy]]",
                      "Mean[mismatch[fancy]]", "Max[mismatch[fancy]]", "Sigma[mm(fancy)]"]])

for i in range(0, len(result)):
    res = np.array([[result[i][0], result[i][1], result[i][2],
                   result[i][3], result[i][4], result[i][5]]])
    output = np.append(output, res, axis=0)

    if output_ON:
        np.savetxt(file_name, output, delimiter='\t', fmt="%s")
