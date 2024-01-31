import os
from pprint import pprint
from pycbc.waveform import get_td_waveform
from pycbc.psd import aLIGODesignSensitivityP1200087, AdVDesignSensitivityP1200087, EinsteinTelescopeP1600143, aLIGOZeroDetHighPower
from pycbc.types.timeseries import TimeSeries
from pycbc.types.frequencyseries import FrequencySeries
from joblib import Parallel, delayed
import gc
import numpy as np
import statistics as stat
import random as rand
import math
import random as rand
import lal
import lalsimulation as lalsim
from scipy.spatial.transform import Rotation
from pycbc.filter import compute_max_snr_over_sky_loc_stat_no_phase, sigmasq, matched_filter_core, overlap_cplx, match
from scipy import optimize
import sys
import matplotlib.pyplot as plt
try:
    import EOBRun_module
except:
    print("No TEOB")
try:

    from pyseobnr.generate_waveform import GenerateWaveform
    from pyseobnr.generate_waveform import generate_prec_hpc_opt
except:
    print("No SEOB in this env")
Pi = math.pi


def nextpow2(x):
    return pow(2, math.ceil(np.log(x)/np.log(2)))


def EOB_par(M, q, f0, s1x, s1y, s1z, s2x, s2y, s2z, distance, inclination):
    return {'M': M,
            'q': q,
            'chi1x': s1x,
            'chi1y': s1y,
            'chi1z': s1z,
            'chi2x': s2x,
            'chi2y': s2y,
            'chi2z': s2z,
            'distance': distance,
            'inclination': inclination,
            'coalescence_angle': 0.0,
            'use_geometric_units': 'no',
            'initial_frequency': f0,
            'domain': 0,
            'interp_uniform_grid': 'yes',
            'srate_interp': 4096.0,
            'use_mode_lm': [0, 1],
            'arg_out': 'no',
            'LambdaAl2': 0,
            'LambdaBl2': 0,
            'output_lm': [1],
            'output_hpc': 'no',
            'output_multipoles': 'no',
            'output_dynamics': 'no',
            'use_spins': 2,
            'postadiabatic_dynamics': 'no',
            'project_spins': 'no',
            'use_flm': 'HM',
            'nqc': 'manual',
            'nqc_coefs_flx': 'nrfit_spin202002',
            'nqc_coefs_hlm': 'none',
            'df': 0.125}


def GenSEOBV5(pars, f_peak):

    params_dict = {}
    params_dict["mass1"] = pars["M"]*pars["q"]/(1.+pars["q"])
    params_dict["mass2"] = pars["M"]/(1.+pars["q"])
    params_dict["spin1x"] = pars["chi1x"]
    params_dict["spin1y"] = pars["chi1y"]
    params_dict["spin1z"] = pars["chi1z"]
    params_dict["spin2x"] = pars["chi2x"]
    params_dict["spin2y"] = pars["chi2y"]
    params_dict["spin2z"] = pars["chi2z"]
    params_dict["deltaT"] = 1/pars["srate_interp"]
    params_dict["f22_start"] = pars["initial_frequency"]
    params_dict["f_ref"]=pars["initial_frequency"]
    params_dict["phi_ref"] = pars["coalescence_angle"]
    params_dict["distance"] = pars["distance"]
    params_dict["inclination"] = pars["inclination"]
    # params_dict["f_max"] = f_peak
    params_dict["mode_array"] = [(2, 2), (2, 1)]
    params_dict["approximant"] = "SEOBNRv5PHM"

    waveform_gen = GenerateWaveform(params_dict)
    # t, hlm = waveform_gen.generate_td_modes()
    hp, hc = waveform_gen.generate_td_polarizations_conditioned_2()
    return hp.deltaT, hp, hc


def GenLALWfFD(pars, approx='IMRPhenomXPHM'):
    params = lal.CreateDict()
    modearr = lalsim.SimInspiralCreateModeArray()
    modes = [(2, 2), (2, -2), (2, 1), (2, -1)]
    for mode in modes:
        lalsim.SimInspiralModeArrayActivateMode(modearr, mode[0], mode[1])
        lalsim.SimInspiralWaveformParamsInsertModeArray(params, modearr)

    q = pars['q']
    M = pars['M']
    c1x, c1y, c1z = pars['chi1x'], pars['chi1y'], pars['chi1z']
    c2x, c2y, c2z = pars['chi2x'], pars['chi2y'], pars['chi2z']
    DL = pars['distance']*1e6*lal.PC_SI
    iota = pars['inclination']
    phir = pars['coalescence_angle']
    df = pars['df']
    flow = pars['initial_frequency']
    srate = pars['srate_interp']
    m1 = M*q/(1.+q)
    m2 = M/(1.+q)
    m1SI = m1*lal.MSUN_SI
    m2SI = m2*lal.MSUN_SI

    app = lalsim.GetApproximantFromString(approx)

    hpf, hcf = lalsim.SimInspiralFD(m1SI, m2SI, c1x, c1y, c1z, c2x, c2y,
                                    c2z, DL, iota, phir, 0., 0., 0., df, flow, srate/2, flow, params, app)
    f = np.array(range(0, len(hpf.data.data)))*hpf.deltaF

    return f, hpf.data.data, hcf.data.data
def GenLALWfFDnomb(pars, approx='IMRPhenomXPHM'):
    params = lal.CreateDict()
    modearr = lalsim.SimInspiralCreateModeArray()
    modes = [(2, 2), (2, -2), (2, 1), (2, -1)]
    for mode in modes:
        lalsim.SimInspiralModeArrayActivateMode(modearr, mode[0], mode[1])
        lalsim.SimInspiralWaveformParamsInsertModeArray(params, modearr)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(params, 0)
    q = pars['q']
    M = pars['M']
    c1x, c1y, c1z = pars['chi1x'], pars['chi1y'], pars['chi1z']
    c2x, c2y, c2z = pars['chi2x'], pars['chi2y'], pars['chi2z']
    DL = pars['distance']*1e6*lal.PC_SI
    iota = pars['inclination']
    phir = pars['coalescence_angle']
    df = pars['df']
    flow = pars['initial_frequency']
    srate = pars['srate_interp']
    m1 = M*q/(1.+q)
    m2 = M/(1.+q)
    m1SI = m1*lal.MSUN_SI
    m2SI = m2*lal.MSUN_SI

    app = lalsim.GetApproximantFromString(approx)

    hpf, hcf = lalsim.SimInspiralFD(m1SI, m2SI, c1x, c1y, c1z, c2x, c2y,
                                    c2z, DL, iota, phir, 0., 0., 0., df, flow, srate/2, flow, params, app)
    f = np.array(range(0, len(hpf.data.data)))*hpf.deltaF

    return f, hpf.data.data, hcf.data.data

def GenLALWfTD(pars, approx='SEOBNRv4P'):
    # create empty LALdict
    params = lal.CreateDict()
    modearr = lalsim.SimInspiralCreateModeArray()
    if approx == "IMRPhenomTPHM":
        modes = [(2, 2), (2, 1), (2, -2), (2, -1)]
    else:
        modes = [(2, 2), (2, 1)]
    for mode in modes:
        lalsim.SimInspiralModeArrayActivateMode(modearr, mode[0], mode[1])
    lalsim.SimInspiralWaveformParamsInsertModeArray(params, modearr)

    # read in from pars
    q = pars['q']
    M = pars['M']
    c1x, c1y, c1z = pars['chi1x'], pars['chi1y'], pars['chi1z']
    c2x, c2y, c2z = pars['chi2x'], pars['chi2y'], pars['chi2z']
    DL = pars['distance']*1e6*lal.PC_SI
    iota = pars['inclination']
    phir = pars['coalescence_angle']
    dT = 1./pars['srate_interp']
    flow = pars['initial_frequency']
    srate = pars['srate_interp']
    # Compute masses
    m1 = M*q/(1.+q)
    m2 = M/(1.+q)
    m1SI = m1*lal.MSUN_SI
    m2SI = m2*lal.MSUN_SI

    app = lalsim.GetApproximantFromString(approx)

    hp, hc = lalsim.SimInspiralTD(m1SI, m2SI, c1x, c1y, c1z, c2x, c2y,
                                  c2z, DL, iota, phir, 0., 0., 0., dT, flow, flow, params, app)
    f = np.array(range(0, len(hp.data.data)))*hp.deltaT

    return f, hp.data.data, hc.data.data


def GenNRsur(pars):
    if pars["q"] < 1:
        pars["q"] = 1/pars["q"]
    m1 = pars["M"]*pars["q"]/(1. + pars["q"])
    m2 = m1/pars["q"]
    hp_nr, hc_nr = get_td_waveform(mass1=m1, mass2=m2, spin1x=pars["chi1x"], spin1y=pars["chi1y"], spin1z=pars["chi1z"],
                                   spin2x=pars["chi2x"], spin2y=pars["chi2y"], spin2z=pars["chi2z"],
                                   eccentricity=0., inclination=pars["inclination"],
                                   coa_phase=pars["coalescence_angle"], distance=pars["distance"], f_lower=0., delta_t=1/pars["srate_interp"],
                                   f_ref=pars["initial_frequency"], mode_array=[(2, 2), (2, 1)], approximant="NRSur7dq4"
                                   )
    return (hp_nr, hc_nr)

def GenNRsur_nom1(pars):
    if pars["q"] < 1:
        pars["q"] = 1/pars["q"]
    m1 = pars["M"]*pars["q"]/(1. + pars["q"])
    m2 = m1/pars["q"]
    hp_nr, hc_nr = get_td_waveform(mass1=m1, mass2=m2, spin1x=pars["chi1x"], spin1y=pars["chi1y"], spin1z=pars["chi1z"],
                                   spin2x=pars["chi2x"], spin2y=pars["chi2y"], spin2z=pars["chi2z"],
                                   eccentricity=0., inclination=pars["inclination"],
                                   coa_phase=pars["coalescence_angle"], distance=pars["distance"], f_lower=0., delta_t=1/pars["srate_interp"],
                                   f_ref=pars["initial_frequency"], mode_array=[(2, 2)], approximant="NRSur7dq4"
                                   )
    return (hp_nr, hc_nr)


def rotate_in_plane_spins(chiA, chiB, theta=0.):

    zaxis = np.array([0, 0, 1])
    r = Rotation.from_rotvec(theta*zaxis)
    chiA_rot = r.apply(chiA)
    chiB_rot = r.apply(chiB)
    return chiA_rot, chiB_rot


def sky_and_time_maxed_match(s, hp, hc, psd, low_freq, high_freq):

    ss = sigmasq(s,  psd=psd, low_frequency_cutoff=low_freq,
                 high_frequency_cutoff=high_freq)
    hphp = sigmasq(hp, psd=psd, low_frequency_cutoff=low_freq,
                   high_frequency_cutoff=high_freq)
    hchc = sigmasq(hc, psd=psd, low_frequency_cutoff=low_freq,
                   high_frequency_cutoff=high_freq)
    hp /= np.sqrt(hphp)
    hc /= np.sqrt(hchc)

    rhop, _, nrm = matched_filter_core(
        hp, s, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    rhop *= nrm
    rhoc, _, nrm = matched_filter_core(
        hc, s, psd=psd, low_frequency_cutoff=low_freq, high_frequency_cutoff=high_freq)
    rhoc *= nrm

    hphccorr = overlap_cplx(hp, hc, psd=psd, low_frequency_cutoff=low_freq,
                            high_frequency_cutoff=high_freq)
    hphccorr = np.real(hphccorr)
    Ipc = hphccorr

    rhop2 = np.abs(rhop.data)**2
    rhoc2 = np.abs(rhoc.data)**2
    gamma = rhop.data * np.conjugate(rhoc.data)
    gamma = np.real(gamma)

    sqrt_part = np.sqrt((rhop2-rhoc2)**2 + 4 *
                        (Ipc*rhop2-gamma)*(Ipc*rhoc2-gamma))
    num = rhop2 - 2.*Ipc*gamma + rhoc2 + sqrt_part
    den = 1. - Ipc**2

    o = np.sqrt(max(num)/den/2.)/np.sqrt(ss)
    if (o > 1.):
        o = 1.

    return o


def dual_annealing(func, bounds, maxfun=500):
    result = optimize.dual_annealing(
        func, bounds, maxfun=maxfun)
    opt_pars, opt_val = result['x'], result['fun']
    return opt_pars, opt_val
def nr_nr_mm(case_number, data, inclination, mr_flag, distance, maxFun, mflag):
    data_number = case_number-1
    q = 1/data["q"][data_number] if data["q"][0] < 1 else data["q"][data_number]
    theta1 = data["theta1"][data_number]
    theta2 = data["theta2"][data_number]
    phi1 = data["phi1"][data_number]
    phi2 = data["phi2"][data_number]
    mass = data["M_tot"][data_number]
    f0 = data["f0"][data_number]
    chi1 = data["chi1"][data_number]
    chi2 = data["chi2"][data_number]
    s1angles = [theta1, phi1]
    s2angles = [theta2, phi2]

    s1x = chi1*np.sin(s1angles[0])*np.cos(s1angles[1])
    s1y = chi1*np.sin(s1angles[0])*np.sin(s1angles[1])
    s1z = chi1*np.cos(s1angles[0])
    s2x = chi2*np.sin(s2angles[0])*np.cos(s2angles[1])
    s2y = chi2*np.sin(s2angles[0])*np.sin(s2angles[1])
    s2z = chi2*np.cos(s2angles[0])

    chiA = [s1x, s1y, s1z]
    chiB = [s2x, s2y, s2z]

    pars_dict = EOB_par(mass, q, f0, s1x, s1y, s1z, s2x,
                        s2y, s2z, distance, inclination)
    dT = 1/pars_dict["srate_interp"]
    if mflag == 0:
        flow = f0+3
    elif mflag==2:
        flow=f0+3
        if mass >= 210: # when flow drops below 11Hz
          flow = 11.0
    else:
        flow = 11.0
    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*data["f_peak"][data_number]
    mm_list = []
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phic in phi_vals:
        pars_dict['coalescence_angle'] = phic
        hp_nr, hc_nr = GenNRsur(pars_dict)
        hp_ts = TimeSeries(hp_nr.data, dT)
        hc_ts = TimeSeries(hc_nr.data, dT)
        hp_eob, hc_eob = GenNRsur_nom1(pars_dict)
        hp_eobTS = TimeSeries(hp_eob, dT)
        hc_eobTS = TimeSeries(hc_eob, dT)

        tl = (max(len(hp_ts), len(hp_eobTS))-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hp_eobTS.resize(tlen)
        hc_eobTS.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()

        hpf_eob = hp_eobTS.to_frequencyseries()
        hcf_eob = hc_eobTS.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for k in kappa_vals:
            if (k == 0.0) and (phic == 0.0):
                m, _ = match(hp_ts, hp_eobTS, psd=psd,
                             low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)
            # h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr
            h = np.cos(k) * hpf_eob + np.sin(k)*hcf_eob

            def localfunc_spinpr(x):
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # gen phenom
                hp_nr, hc_nr = GenNRsur(pars_dict)
                hp_TS = TimeSeries(hp_nr, dT)
                hc_TS = TimeSeries(hc_nr, dT)
                hp_TS.resize(tlen)
                hc_TS.resize(tlen)
                hpf = hp_TS.to_frequencyseries()
                hcf = hc_TS.to_frequencyseries()
                ans = 1. - \
                    sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)
                return ans

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(case_number, mass, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [case_number, 1.-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]

def nr_xphm_mm(case_number, data, inclination, mr_flag, distance, maxFun, mflag):
    data_number = case_number-1
    q = 1/data["q"][data_number] if data["q"][0] < 1 else data["q"][data_number]
    theta1 = data["theta1"][data_number]
    theta2 = data["theta2"][data_number]
    phi1 = data["phi1"][data_number]
    phi2 = data["phi2"][data_number]
    mass = data["M_tot"][data_number]
    f0 = data["f0"][data_number]
    chi1 = data["chi1"][data_number]
    chi2 = data["chi2"][data_number]
    s1angles = [theta1, phi1]
    s2angles = [theta2, phi2]

    s1x = chi1*np.sin(s1angles[0])*np.cos(s1angles[1])
    s1y = chi1*np.sin(s1angles[0])*np.sin(s1angles[1])
    s1z = chi1*np.cos(s1angles[0])
    s2x = chi2*np.sin(s2angles[0])*np.cos(s2angles[1])
    s2y = chi2*np.sin(s2angles[0])*np.sin(s2angles[1])
    s2z = chi2*np.cos(s2angles[0])

    chiA = [s1x, s1y, s1z]
    chiB = [s2x, s2y, s2z]

    pars_dict = EOB_par(mass, q, f0, s1x, s1y, s1z, s2x,
                        s2y, s2z, distance, inclination)
    dT = 1/pars_dict["srate_interp"]
    if mflag == 0:
        flow = f0+3
    elif mflag==2:
        flow=f0+3
        if mass >= 210: # when flow drops below 11Hz
          flow = 11.0
    else:
        flow = 11.0
    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*data["f_peak"][data_number]
    mm_list = []
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phic in phi_vals:
        pars_dict['coalescence_angle'] = phic
        hp_nr, hc_nr = GenNRsur(pars_dict)
        hp_ts = TimeSeries(hp_nr.data, dT)
        hc_ts = TimeSeries(hc_nr.data, dT)
        tl = (len(hp_ts)-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for k in kappa_vals:
            h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr
            f, hp_ph, hc_ph = GenLALWfFD(pars_dict)
            hpf = FrequencySeries(hp_ph, f[1]-f[0])
            hcf = FrequencySeries(hc_ph, f[1]-f[0])
            if (k == 0.0) and (phic == 0.0):
                m,  _ = match(hp_ts, hpf, psd=psd,
                              low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)

            def localfunc_spinpr(x):
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # gen phenom
                f, hp_ph, hc_ph = GenLALWfFD(pars_dict)
                hpf = FrequencySeries(hp_ph, f[1]-f[0])
                hcf = FrequencySeries(hc_ph, f[1]-f[0])

                return 1.-sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(case_number, mass, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [case_number, 1.-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]

def nr_xphm_nomb_mm(case_number, data, inclination, mr_flag, distance, maxFun, mflag):
    data_number = case_number-1
    q = 1/data["q"][data_number] if data["q"][0] < 1 else data["q"][data_number]
    theta1 = data["theta1"][data_number]
    theta2 = data["theta2"][data_number]
    phi1 = data["phi1"][data_number]
    phi2 = data["phi2"][data_number]
    mass = data["M_tot"][data_number]
    f0 = data["f0"][data_number]
    chi1 = data["chi1"][data_number]
    chi2 = data["chi2"][data_number]
    s1angles = [theta1, phi1]
    s2angles = [theta2, phi2]

    s1x = chi1*np.sin(s1angles[0])*np.cos(s1angles[1])
    s1y = chi1*np.sin(s1angles[0])*np.sin(s1angles[1])
    s1z = chi1*np.cos(s1angles[0])
    s2x = chi2*np.sin(s2angles[0])*np.cos(s2angles[1])
    s2y = chi2*np.sin(s2angles[0])*np.sin(s2angles[1])
    s2z = chi2*np.cos(s2angles[0])

    chiA = [s1x, s1y, s1z]
    chiB = [s2x, s2y, s2z]

    pars_dict = EOB_par(mass, q, f0, s1x, s1y, s1z, s2x,
                        s2y, s2z, distance, inclination)
    dT = 1/pars_dict["srate_interp"]
    if mflag == 0:
        flow = f0+3
    elif mflag==2:
        flow=f0+3
        if mass >= 210: # when flow drops below 11Hz
          flow = 11.0
    else:
        flow = 11.0
    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*data["f_peak"][data_number]
    mm_list = []
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phic in phi_vals:
        pars_dict['coalescence_angle'] = phic
        hp_nr, hc_nr = GenNRsur(pars_dict)
        hp_ts = TimeSeries(hp_nr.data, dT)
        hc_ts = TimeSeries(hc_nr.data, dT)
        tl = (len(hp_ts)-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for k in kappa_vals:
            h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr
            f, hp_ph, hc_ph =GenLALWfFDnomb(pars_dict)
            hpf = FrequencySeries(hp_ph, f[1]-f[0])
            hcf = FrequencySeries(hc_ph, f[1]-f[0])
            if (k == 0.0) and (phic == 0.0):
                m,  _ = match(hp_ts, hpf, psd=psd,
                              low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)

            def localfunc_spinpr(x):
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # gen phenom
                f, hp_ph, hc_ph = GenLALWfFDnomb(pars_dict)
                hpf = FrequencySeries(hp_ph, f[1]-f[0])
                hcf = FrequencySeries(hc_ph, f[1]-f[0])

                return 1.-sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(case_number, mass, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [case_number, 1.-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]
def nr_teob_mm(case_number, data, inclination, mr_flag, distance, maxFun, mflag):
    data_number = case_number-1
    q = 1/data["q"][data_number] if data["q"][0] < 1 else data["q"][data_number]
    theta1 = data["theta1"][data_number]
    theta2 = data["theta2"][data_number]
    phi1 = data["phi1"][data_number]
    phi2 = data["phi2"][data_number]
    mass = data["M_tot"][data_number]
    f0 = data["f0"][data_number]
    chi1 = data["chi1"][data_number]
    chi2 = data["chi2"][data_number]
    s1angles = [theta1, phi1]
    s2angles = [theta2, phi2]

    s1x = chi1*np.sin(s1angles[0])*np.cos(s1angles[1])
    s1y = chi1*np.sin(s1angles[0])*np.sin(s1angles[1])
    s1z = chi1*np.cos(s1angles[0])
    s2x = chi2*np.sin(s2angles[0])*np.cos(s2angles[1])
    s2y = chi2*np.sin(s2angles[0])*np.sin(s2angles[1])
    s2z = chi2*np.cos(s2angles[0])

    chiA = [s1x, s1y, s1z]
    chiB = [s2x, s2y, s2z]

    pars_dict = EOB_par(mass, q, f0, s1x, s1y, s1z, s2x,
                        s2y, s2z, distance, inclination)
    dT = 1/pars_dict["srate_interp"]
    if mflag == 0:
        flow = f0+3
    elif mflag==2:
        flow=f0+3
        if mass >= 210: # when flow drops below 11Hz
          flow = 11.0
    else:
        flow = 11.0
    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*data["f_peak"][data_number]
    mm_list = []
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phic in phi_vals:
        pars_dict['coalescence_angle'] = phic
        hp_nr, hc_nr = GenNRsur(pars_dict)
        hp_ts = TimeSeries(hp_nr.data, dT)
        hc_ts = TimeSeries(hc_nr.data, dT)
        t, hp_eob, hc_eob = EOBRun_module.EOBRunPy(pars_dict)
        hp_eobTS = TimeSeries(hp_eob, dT)
        hc_eobTS = TimeSeries(hc_eob, dT)

        tl = (max(len(hp_ts), len(hp_eobTS))-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hp_eobTS.resize(tlen)
        hc_eobTS.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()

        hpf_eob = hp_eobTS.to_frequencyseries()
        hcf_eob = hc_eobTS.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for k in kappa_vals:
            if (k == 0.0) and (phic == 0.0):
                m, _ = match(hp_ts, hp_eobTS, psd=psd,
                             low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)
            # h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr
            h = np.cos(k) * hpf_eob + np.sin(k)*hcf_eob

            def localfunc_spinpr(x):
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # gen phenom
                hp_nr, hc_nr = GenNRsur(pars_dict)
                hp_TS = TimeSeries(hp_nr, dT)
                hc_TS = TimeSeries(hc_nr, dT)
                hp_TS.resize(tlen)
                hc_TS.resize(tlen)
                hpf = hp_TS.to_frequencyseries()
                hcf = hc_TS.to_frequencyseries()
                ans = 1. - \
                    sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)
                return ans

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(case_number, mass, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [case_number, 1.-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]


def nr_seob_mm(case_number, data, inclination, mr_flag, distance, maxFun, mflag):
    data_number = case_number-1
    q = 1/data["q"][data_number] if data["q"][0] < 1 else data["q"][data_number]
    theta1 = data["theta1"][data_number]
    theta2 = data["theta2"][data_number]
    phi1 = data["phi1"][data_number]
    phi2 = data["phi2"][data_number]
    mass = data["M_tot"][data_number]
    f0 = data["f0"][data_number]
    chi1 = data["chi1"][data_number]
    chi2 = data["chi2"][data_number]
    s1angles = [theta1, phi1]
    s2angles = [theta2, phi2]

    s1x = chi1*np.sin(s1angles[0])*np.cos(s1angles[1])
    s1y = chi1*np.sin(s1angles[0])*np.sin(s1angles[1])
    s1z = chi1*np.cos(s1angles[0])
    s2x = chi2*np.sin(s2angles[0])*np.cos(s2angles[1])
    s2y = chi2*np.sin(s2angles[0])*np.sin(s2angles[1])
    s2z = chi2*np.cos(s2angles[0])

    chiA = [s1x, s1y, s1z]
    chiB = [s2x, s2y, s2z]

    pars_dict = EOB_par(mass, q, f0, s1x, s1y, s1z, s2x,
                        s2y, s2z, distance, inclination)
    dT = 1/pars_dict["srate_interp"]
    if mflag == 0:
        flow = f0+3
    elif mflag==2:
        flow=f0+3
        if mass >= 210: # when flow drops below 11Hz
          flow = 11.0
    else:
        flow = 11.0
    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*data["f_peak"][data_number]
    mm_list = []
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phic in phi_vals:
        pars_dict['coalescence_angle'] = phic
        hp_nr, hc_nr = GenNRsur(pars_dict)
        hp_ts = TimeSeries(hp_nr.data, dT)
        hc_ts = TimeSeries(hc_nr.data, dT)
        t, hp_eob, hc_eob = GenLALWfTD(pars_dict, approx='SEOBNRv4P')
        hp_eobTS = TimeSeries(hp_eob, dT)
        hc_eobTS = TimeSeries(hc_eob, dT)

        tl = (max(len(hp_ts), len(hp_eobTS))-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hp_eobTS.resize(tlen)
        hc_eobTS.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()

        hpf_eob = hp_eobTS.to_frequencyseries()
        hcf_eob = hc_eobTS.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for k in kappa_vals:
            # h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr
            h = np.cos(k) * hpf_eob + np.sin(k)*hcf_eob
            if (k == 0.0) and (phic == 0.0):
                m, _ = match(hp_ts, hp_eobTS, psd=psd,
                             low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)

            def localfunc_spinpr(x):
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # gen phenom
                hp_nr, hc_nr = GenNRsur(pars_dict)
                hp_TS = TimeSeries(hp_nr, dT)
                hc_TS = TimeSeries(hc_nr, dT)
                hp_TS.resize(tlen)
                hc_TS.resize(tlen)
                hpf = hp_TS.to_frequencyseries()
                hcf = hc_TS.to_frequencyseries()
                return 1.-sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(case_number, mass, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [case_number, 1.-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]


def nr_seobV5_mm(case_number, data, inclination, mr_flag, distance, maxFun, mflag):
    data_number = case_number-1
    q = 1/data["q"][data_number] if data["q"][0] < 1 else data["q"][data_number]
    theta1 = data["theta1"][data_number]
    theta2 = data["theta2"][data_number]
    phi1 = data["phi1"][data_number]
    phi2 = data["phi2"][data_number]
    mass = data["M_tot"][data_number]
    f0 = data["f0"][data_number]
    chi1 = data["chi1"][data_number]
    chi2 = data["chi2"][data_number]
    s1angles = [theta1, phi1]
    s2angles = [theta2, phi2]

    s1x = chi1*np.sin(s1angles[0])*np.cos(s1angles[1])
    s1y = chi1*np.sin(s1angles[0])*np.sin(s1angles[1])
    s1z = chi1*np.cos(s1angles[0])
    s2x = chi2*np.sin(s2angles[0])*np.cos(s2angles[1])
    s2y = chi2*np.sin(s2angles[0])*np.sin(s2angles[1])
    s2z = chi2*np.cos(s2angles[0])

    chiA = [s1x, s1y, s1z]
    chiB = [s2x, s2y, s2z]

    pars_dict = EOB_par(mass, q, f0, s1x, s1y, s1z, s2x,
                        s2y, s2z, distance, inclination)
    dT = 1/pars_dict["srate_interp"]
    if mflag == 0:
        flow = f0+3
    elif mflag==2:
        flow=f0+3
        if mass >= 210: # when flow drops below 11Hz
          flow = 11.0
    else:
        flow = 11.0
    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*data["f_peak"][data_number]
    mm_list = []
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phic in phi_vals:
        pars_dict['coalescence_angle'] = phic
        hp_nr, hc_nr = GenNRsur(pars_dict)
        hp_ts = TimeSeries(hp_nr.data, dT)
        hc_ts = TimeSeries(hc_nr.data, dT)
        t, hp_eob, hc_eob = GenSEOBV5(pars_dict, fhigh)
        hp_eobTS = TimeSeries(hp_eob.data.data, dT)
        hc_eobTS = TimeSeries(hc_eob.data.data, dT)

        tl = (max(len(hp_ts), len(hp_eobTS))-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hp_eobTS.resize(tlen)
        hc_eobTS.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()

        hpf_eob = hp_eobTS.to_frequencyseries()
        hcf_eob = hc_eobTS.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for k in kappa_vals:
            # h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr
            h = np.cos(k) * hpf_eob + np.sin(k)*hcf_eob
            if (k == 0.0) and (phic == 0.0):
                m, _ = match(hp_ts, hp_eobTS, psd=psd,
                             low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)

            def localfunc_spinpr(x):
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # gen phenom
                hp_nr, hc_nr = GenNRsur(pars_dict)
                hp_TS = TimeSeries(hp_nr, dT)
                hc_TS = TimeSeries(hc_nr, dT)
                hp_TS.resize(tlen)
                hc_TS.resize(tlen)
                hpf = hp_TS.to_frequencyseries()
                hcf = hc_TS.to_frequencyseries()
                return 1.-sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(case_number, mass, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [case_number, 1.-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]


def nr_tphm_mm(case_number, data, inclination, mr_flag, distance, maxFun, mflag):
    data_number = case_number-1
    q = 1/data["q"][data_number] if data["q"][0] < 1 else data["q"][data_number]
    theta1 = data["theta1"][data_number]
    theta2 = data["theta2"][data_number]
    phi1 = data["phi1"][data_number]
    phi2 = data["phi2"][data_number]
    mass = data["M_tot"][data_number]
    f0 = data["f0"][data_number]
    chi1 = data["chi1"][data_number]
    chi2 = data["chi2"][data_number]
    s1angles = [theta1, phi1]
    s2angles = [theta2, phi2]

    s1x = chi1*np.sin(s1angles[0])*np.cos(s1angles[1])
    s1y = chi1*np.sin(s1angles[0])*np.sin(s1angles[1])
    s1z = chi1*np.cos(s1angles[0])
    s2x = chi2*np.sin(s2angles[0])*np.cos(s2angles[1])
    s2y = chi2*np.sin(s2angles[0])*np.sin(s2angles[1])
    s2z = chi2*np.cos(s2angles[0])

    chiA = [s1x, s1y, s1z]
    chiB = [s2x, s2y, s2z]

    pars_dict = EOB_par(mass, q, f0, s1x, s1y, s1z, s2x,
                        s2y, s2z, distance, inclination)
    dT = 1/pars_dict["srate_interp"]
    if mflag == 0:
        flow = f0+3
    elif mflag==2:
        flow=f0+3
        if mass >= 210: # when flow drops below 11Hz
          flow = 11.0
    else:
        flow = 11.0
    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*data["f_peak"][data_number]
    mm_list = []
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phic in phi_vals:
        pars_dict['coalescence_angle'] = phic
        hp_nr, hc_nr = GenNRsur(pars_dict)
        hp_ts = TimeSeries(hp_nr.data, dT)
        hc_ts = TimeSeries(hc_nr.data, dT)
        t, hp_eob, hc_eob = GenLALWfTD(pars_dict, approx="IMRPhenomTPHM")
        hp_eobTS = TimeSeries(hp_eob, dT)
        hc_eobTS = TimeSeries(hc_eob, dT)

        tl = (max(len(hp_ts), len(hp_eobTS))-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hp_eobTS.resize(tlen)
        hc_eobTS.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for k in kappa_vals:

            h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr
            if (k == 0.0) and (phic == 0.0):
                m, _ = match(hp_ts, hp_eobTS, psd=psd,
                             low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)

            def localfunc_spinpr(x):
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # gen phenom
                t, hp_eob, hc_eob = GenLALWfTD(
                    pars_dict, approx="IMRPhenomTPHM")
                hp_eobTS = TimeSeries(hp_eob, dT)
                hc_eobTS = TimeSeries(hc_eob, dT)
                hp_eobTS.resize(tlen)
                hc_eobTS.resize(tlen)
                hpf = hp_eobTS.to_frequencyseries()
                hcf = hc_eobTS.to_frequencyseries()
                return 1.-sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(case_number, mass, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [case_number, 1.-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]


def ReadSXSWithLAL(fname, M, iota, dT, DL=1., phi_ref=0., modes=[(2, 2), (2, -2), (2, 1), (2, -1), (2, 0)]):

    import h5py
    try:
        import lalsimulation as lalsim
        import lal
    except Exception:
        print("Error importing LALSimulation and/or LAL")

    # create empty LALdict
    params = lal.CreateDict()

    # read h5
    f = h5py.File(fname, 'r')

    # Modes (default (2,2) only)
    modearr = lalsim.SimInspiralCreateModeArray()
    for mode in modes:
        lalsim.SimInspiralModeArrayActivateMode(modearr, mode[0], mode[1])
    lalsim.SimInspiralWaveformParamsInsertModeArray(params, modearr)
    lalsim.SimInspiralWaveformParamsInsertNumRelData(params, fname)

    Mt = f.attrs['mass1'] + f.attrs['mass2']
    m1 = f.attrs['mass1']*M/Mt
    m2 = f.attrs['mass2']*M/Mt
    m1SI = m1 * lal.MSUN_SI
    m2SI = m2 * lal.MSUN_SI
    DLmpc = DL*1e6*lal.PC_SI  # assuming DL given in in Mpc

    flow = f.attrs['f_lower_at_1MSUN']/M
    fref = flow

    spins = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(fref, M, fname)
    c1x, c1y, c1z = spins[0], spins[1], spins[2]
    c2x, c2y, c2z = spins[3], spins[4], spins[5]

    new_params = {}
    new_params['m1'] = m1
    new_params['m2'] = m2
    new_params['chiA'] = [c1x, c1y, c1z]
    new_params['chiB'] = [c2x, c2y, c2z]
    new_params['flow'] = flow

    if (0):
        print("chiA=", c1x, c1y, c1z)
        print("chiB=", c2x, c2y, c2z)

    hp, hc = lalsim.SimInspiralChooseTDWaveform(
        m1SI, m2SI, c1x, c1y, c1z, c2x, c2y, c2z, DLmpc, iota, phi_ref, 0., 0., 0., dT, flow, fref, params, lalsim.NR_hdf5)

    return hp, hc, new_params


def sxs_xphm_mm(case_number, sxs_data, inclination, mr_flag, distance, maxFun, mflag, fpeak_data):
    simNo = sxs_data[:, 0]
    simResLev = sxs_data[:, 3]
    k = case_number
    Ni = round(simNo[k])
    if Ni < 10:
        N = "000" + str(Ni)
    elif Ni < 100:
        N = "00" + str(Ni)
    elif Ni < 1000:
        N = "0" + str(Ni)
    else:
        N = str(Ni)

    # print("SXS:",N)
    fgw = (
        "data/SXS/SXS_BBH_"
        + N
        + "_Res"
        + str(round(simResLev[k]))
        + ".h5"
    )

    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*fpeak_data[case_number]
    mm_list = []
    if mflag == 0:
        M = 37.5
    elif mflag == 1:
        M = 150
    dT = 1.0 / (4096)
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phi in phi_vals:
        phic = phi
        hp_nr, hc_nr, pars = ReadSXSWithLAL(
            fgw, M, inclination, dT, distance, phic)
        m1 = pars["m1"]
        m2 = pars["m2"]
        c1x, c1y, c1z = chiA = pars["chiA"]
        c2x, c2y, c2z = chiB = pars["chiB"]
        f0 = pars["flow"]
        flow = f0+3
        pars_dict = EOB_par(M, m1/m2, f0, c1x, c1y, c1z, c2x,
                            c2y, c2z, distance, inclination)

        pars_dict['coalescence_angle'] = phic
        hp_ts = TimeSeries(hp_nr.data.data, dT)
        hc_ts = TimeSeries(hc_nr.data.data, dT)
        tl = (len(hp_ts)-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for kappa in kappa_vals:
            k = kappa
            h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr

            f, hp_ph, hc_ph = GenLALWfFD(pars_dict)
            hpf = FrequencySeries(hp_ph, f[1]-f[0])
            hcf = FrequencySeries(hc_ph, f[1]-f[0])
            if (kappa == 0.0) and (phi == 0.):
                m,  _ = match(hp_ts, hpf, psd=psd,
                              low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)

            def localfunc_spinpr(x):
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # gen phenom
                f, hp_ph, hc_ph = GenLALWfFD(pars_dict)
                hpf = FrequencySeries(hp_ph, f[1]-f[0])
                hcf = FrequencySeries(hc_ph, f[1]-f[0])

                return 1.-sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(Ni, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [Ni, 1-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]


def sxs_teob_mm(case_number, sxs_data, inclination, mr_flag, distance, maxFun, mflag, fpeak_data):
    data_number = case_number-1
    simNo = sxs_data[:, 0]
    simResLev = sxs_data[:, 3]
    k = case_number
    Ni = round(simNo[k])
    if Ni < 10:
        N = "000" + str(Ni)
    elif Ni < 100:
        N = "00" + str(Ni)
    elif Ni < 1000:
        N = "0" + str(Ni)
    else:
        N = str(Ni)

    # print("SXS:",N)
    fgw = (
        "data/SXS/SXS_BBH_"
        + N
        + "_Res"
        + str(round(simResLev[k]))
        + ".h5"
    )
    dT = 1.0 / (4096)
    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*fpeak_data[case_number]
    mm_list = []
    if mflag == 0:
        M = 37.5
    elif mflag == 1:
        M = 150
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phi in phi_vals:
        phic = phi
        hp_nr, hc_nr, pars = ReadSXSWithLAL(
            fgw, M, inclination, dT, distance, phic)
        m1 = pars["m1"]
        m2 = pars["m2"]
        c1x, c1y, c1z = chiA = pars["chiA"]
        c2x, c2y, c2z = chiB = pars["chiB"]
        f0 = pars["flow"]
        flow = f0+3
        pars_dict = EOB_par(M, m1/m2, f0, c1x, c1y, c1z, c2x,
                            c2y, c2z, distance, inclination)
        phic = phi
        pars_dict['coalescence_angle'] = phic
        hp_ts = TimeSeries(hp_nr.data.data, dT)
        hc_ts = TimeSeries(hc_nr.data.data, dT)

        t, hp_eob, hc_eob = EOBRun_module.EOBRunPy(pars_dict)
        hp_eobTS = TimeSeries(hp_eob, dT)
        hc_eobTS = TimeSeries(hc_eob, dT)
        tl = (max(len(hp_ts), len(hp_eobTS))-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hp_eobTS.resize(tlen)
        hc_eobTS.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()
        hpf_eob = hp_eobTS.to_frequencyseries()
        hcf_eob = hc_eobTS.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for kappa in kappa_vals:
            k = kappa
            h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr
            if (kappa == 0.0) and (phi == 0.0):
                m,  _ = match(hp_ts, hp_eobTS, psd=psd,
                              low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)

            def localfunc_spinpr(x):  # Anneal over approx :(
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # EOB wf
                t, hp_eob, hc_eob = EOBRun_module.EOBRunPy(pars_dict)
                hp_eobTS = TimeSeries(hp_eob, dT)
                hc_eobTS = TimeSeries(hc_eob, dT)
                hp_eobTS.resize(tlen)
                hc_eobTS.resize(tlen)
                hpf = hp_eobTS.to_frequencyseries()
                hcf = hc_eobTS.to_frequencyseries()
                ans = 1. - \
                    sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)
                return ans

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(Ni, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [Ni, 1-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]


def sxs_seob_mm(case_number, sxs_data, inclination, mr_flag, distance, maxFun, mflag, fpeak_data):
    data_number = case_number-1
    simNo = sxs_data[:, 0]
    simResLev = sxs_data[:, 3]
    k = case_number
    Ni = round(simNo[k])
    if Ni < 10:
        N = "000" + str(Ni)
    elif Ni < 100:
        N = "00" + str(Ni)
    elif Ni < 1000:
        N = "0" + str(Ni)
    else:
        N = str(Ni)

    # print("SXS:",N)
    fgw = (
        "data/SXS/SXS_BBH_"
        + N
        + "_Res"
        + str(round(simResLev[k]))
        + ".h5"
    )

    dT = 1.0 / (4096)
    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*fpeak_data[case_number]
    mm_list = []
    if mflag == 0:
        M = 37.5
    elif mflag == 1:
        M = 150
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phi in phi_vals:
        phic = phi
        hp_nr, hc_nr, pars = ReadSXSWithLAL(
            fgw, M, inclination, dT, distance, phic)
        m1 = pars["m1"]
        m2 = pars["m2"]
        c1x, c1y, c1z = chiA = pars["chiA"]
        c2x, c2y, c2z = chiB = pars["chiB"]
        f0 = pars["flow"]
        flow = f0 + 3.0
        pars_dict = EOB_par(M, m1/m2, f0, c1x, c1y, c1z, c2x,
                            c2y, c2z, distance, inclination)

        pars_dict['coalescence_angle'] = phic
        hp_ts = TimeSeries(hp_nr.data.data, dT)
        hc_ts = TimeSeries(hc_nr.data.data, dT)

        t, hp_eob, hc_eob = GenLALWfTD(pars_dict, approx='SEOBNRv4P')
        hp_eobTS = TimeSeries(hp_eob, dT)
        hc_eobTS = TimeSeries(hc_eob, dT)
        tl = (max(len(hp_ts), len(hp_eobTS))-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hp_eobTS.resize(tlen)
        hc_eobTS.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()
        hpf_eob = hp_eobTS.to_frequencyseries()
        hcf_eob = hc_eobTS.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for kappa in kappa_vals:
            k = kappa
            h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr
            if (kappa == 0.0) and (phi == 0.0):
                m,  _ = match(hp_ts, hp_eobTS, psd=psd,
                              low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)

            def localfunc_spinpr(x):  # Anneal over approx :(
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # EOB wf
                t, hp_eob, hc_eob = GenLALWfTD(pars_dict, approx='SEOBNRv4P')
                hp_eobTS = TimeSeries(hp_eob, dT)
                hc_eobTS = TimeSeries(hc_eob, dT)
                hp_eobTS.resize(tlen)
                hc_eobTS.resize(tlen)
                hpf = hp_eobTS.to_frequencyseries()
                hcf = hc_eobTS.to_frequencyseries()
                ans = 1. - \
                    sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)
                return ans

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(Ni, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [Ni, 1-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]


def sxs_seobV5_mm(case_number, sxs_data, inclination, mr_flag, distance, maxFun, mflag, fpeak_data):
    data_number = case_number-1
    simNo = sxs_data[:, 0]
    simResLev = sxs_data[:, 3]
    k = case_number
    Ni = round(simNo[k])
    if Ni < 10:
        N = "000" + str(Ni)
    elif Ni < 100:
        N = "00" + str(Ni)
    elif Ni < 1000:
        N = "0" + str(Ni)
    else:
        N = str(Ni)

    # print("SXS:",N)
    fgw = (
        "data/SXS/SXS_BBH_"
        + N
        + "_Res"
        + str(round(simResLev[k]))
        + ".h5"
    )

    dT = 1.0 / (4096)
    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*fpeak_data[case_number]
    mm_list = []
    if mflag == 0:
        M = 37.5
    elif mflag == 1:
        M = 150
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phi in phi_vals:
        phic = phi
        hp_nr, hc_nr, pars = ReadSXSWithLAL(
            fgw, M, inclination, dT, distance, phic)
        m1 = pars["m1"]
        m2 = pars["m2"]
        c1x, c1y, c1z = chiA = pars["chiA"]
        c2x, c2y, c2z = chiB = pars["chiB"]
        f0 = pars["flow"]
        flow = f0 + 3.0
        pars_dict = EOB_par(M, m1/m2, f0, c1x, c1y, c1z, c2x,
                            c2y, c2z, distance, inclination)

        pars_dict['coalescence_angle'] = phic
        hp_ts = TimeSeries(hp_nr.data.data, dT)
        hc_ts = TimeSeries(hc_nr.data.data, dT)

        t, hp_eob, hc_eob = GenSEOBV5(pars_dict, fhigh)
        hp_eobTS = TimeSeries(hp_eob.data.data, dT)
        hc_eobTS = TimeSeries(hc_eob.data.data, dT)
        tl = (max(len(hp_ts), len(hp_eobTS))-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hp_eobTS.resize(tlen)
        hc_eobTS.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()
        hpf_eob = hp_eobTS.to_frequencyseries()
        hcf_eob = hc_eobTS.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for kappa in kappa_vals:
            k = kappa
            h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr
            if (kappa == 0.0) and (phi == 0.0):
                m,  _ = match(hp_ts, hp_eobTS, psd=psd,
                              low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)

            def localfunc_spinpr(x):  # Anneal over approx :(
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # EOB wf
                t, hp_eob, hc_eob = GenSEOBV5(pars_dict, fhigh)
                hp_eobTS = TimeSeries(hp_eob.data.data, dT)
                hc_eobTS = TimeSeries(hc_eob.data.data, dT)
                hp_eobTS.resize(tlen)
                hc_eobTS.resize(tlen)
                hpf = hp_eobTS.to_frequencyseries()
                hcf = hc_eobTS.to_frequencyseries()
                ans = 1. - \
                    sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)
                return ans

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(Ni, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [Ni, 1-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]


def sxs_tphm_mm(case_number, sxs_data, inclination, mr_flag, distance, maxFun, mflag, fpeak_data):
    data_number = case_number-1
    simNo = sxs_data[:, 0]
    simResLev = sxs_data[:, 3]
    k = case_number
    Ni = round(simNo[k])
    if Ni < 10:
        N = "000" + str(Ni)
    elif Ni < 100:
        N = "00" + str(Ni)
    elif Ni < 1000:
        N = "0" + str(Ni)
    else:
        N = str(Ni)

    # print("SXS:",N)
    fgw = (
        "data/SXS/SXS_BBH_"
        + N
        + "_Res"
        + str(round(simResLev[k]))
        + ".h5"
    )

    dT = 1.0 / (4096)
    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*fpeak_data[case_number]
    mm_list = []
    if mflag == 0:
        M = 37.5
    elif mflag == 1:
        M = 150
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phi in phi_vals:
        phic = phi
        hp_nr, hc_nr, pars = ReadSXSWithLAL(
            fgw, M, inclination, dT, distance, phic)
        m1 = pars["m1"]
        m2 = pars["m2"]
        c1x, c1y, c1z = chiA = pars["chiA"]
        c2x, c2y, c2z = chiB = pars["chiB"]
        f0 = pars["flow"]
        flow = f0 + 3.0
        pars_dict = EOB_par(M, m1/m2, f0, c1x, c1y, c1z, c2x,
                            c2y, c2z, distance, inclination)
        phic = phi
        pars_dict['coalescence_angle'] = phic
        hp_ts = TimeSeries(hp_nr.data.data, dT)
        hc_ts = TimeSeries(hc_nr.data.data, dT)

        t, hp_eob, hc_eob = GenLALWfTD(pars_dict, approx="IMRPhenomTPHM")
        hp_eobTS = TimeSeries(hp_eob, dT)
        hc_eobTS = TimeSeries(hc_eob, dT)
        tl = (max(len(hp_ts), len(hp_eobTS))-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hp_eobTS.resize(tlen)
        hc_eobTS.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()
        hpf_eob = hp_eobTS.to_frequencyseries()
        hcf_eob = hc_eobTS.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for kappa in kappa_vals:
            k = kappa
            h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr
            if (kappa == 0.0) and (phi == 0.0):
                m,  _ = match(hp_ts, hp_eobTS, psd=psd,
                              low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)

            def localfunc_spinpr(x):  # Anneal over approx :(
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # EOB wf
                t, hp_eob, hc_eob = GenLALWfTD(
                    pars_dict, approx="IMRPhenomTPHM")
                hp_eobTS = TimeSeries(hp_eob, dT)
                hc_eobTS = TimeSeries(hc_eob, dT)
                hp_eobTS.resize(tlen)
                hc_eobTS.resize(tlen)
                hpf = hp_eobTS.to_frequencyseries()
                hcf = hc_eobTS.to_frequencyseries()
                ans = 1. - \
                    sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)
                return ans

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(Ni, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [Ni, 1-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]




def sxs_long_xphm_mm(case_number, sxs_data, inclination, mr_flag, distance, maxFun, mflag, fpeak_data):
    simNo = sxs_data[:, 0]
    simResLev = sxs_data[:, 3]
    k = case_number
    Ni = round(simNo[k])
    if Ni < 10:
        N = "000" + str(Ni)
    elif Ni < 100:
        N = "00" + str(Ni)
    elif Ni < 1000:
        N = "0" + str(Ni)
    else:
        N = str(Ni)

    # print("SXS:",N)
    fgw = (
        "data/SXS/SXS_BBH_"
        + N
        + "_Res"
        + str(round(simResLev[k]))
        + ".h5"
    )

    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*fpeak_data[case_number]
    mm_list = []
    if mflag == 0:
        M = 37.5
    elif mflag == 1:
        M = 150
    elif mflag==2:
        M = 75
    elif mflag==3:
        M = 112.5
    elif mflag==4:
        M = 187.5
    elif mflag==5:
        M = 225
    dT = 1.0 / (4096)
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phi in phi_vals:
        phic = phi
        hp_nr, hc_nr, pars = ReadSXSWithLAL(
            fgw, M, inclination, dT, distance, phic)
        m1 = pars["m1"]
        m2 = pars["m2"]
        c1x, c1y, c1z = chiA = pars["chiA"]
        c2x, c2y, c2z = chiB = pars["chiB"]
        f0 = pars["flow"]
        # if M==37.5:
        #     flow = f0+3
        # else:
        #     flow=11
        flow = f0+3
        pars_dict = EOB_par(M, m1/m2, f0, c1x, c1y, c1z, c2x,
                            c2y, c2z, distance, inclination)

        pars_dict['coalescence_angle'] = phic
        hp_ts = TimeSeries(hp_nr.data.data, dT)
        hc_ts = TimeSeries(hc_nr.data.data, dT)
        tl = (len(hp_ts)-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for kappa in kappa_vals:
            k = kappa
            h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr

            f, hp_ph, hc_ph = GenLALWfFD(pars_dict)
            hpf = FrequencySeries(hp_ph, f[1]-f[0])
            hcf = FrequencySeries(hc_ph, f[1]-f[0])
            if (kappa == 0.0) and (phi == 0.):
                m,  _ = match(hp_ts, hpf, psd=psd,
                              low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)

            def localfunc_spinpr(x):
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # gen phenom
                f, hp_ph, hc_ph = GenLALWfFD(pars_dict)
                hpf = FrequencySeries(hp_ph, f[1]-f[0])
                hcf = FrequencySeries(hc_ph, f[1]-f[0])

                return 1.-sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(Ni, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [Ni, 1-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]


def sxs_long_teob_mm(case_number, sxs_data, inclination, mr_flag, distance, maxFun, mflag, fpeak_data):
    data_number = case_number-1
    simNo = sxs_data[:, 0]
    simResLev = sxs_data[:, 3]
    k = case_number
    Ni = round(simNo[k])
    if Ni < 10:
        N = "000" + str(Ni)
    elif Ni < 100:
        N = "00" + str(Ni)
    elif Ni < 1000:
        N = "0" + str(Ni)
    else:
        N = str(Ni)

    # print("SXS:",N)
    fgw = (
        "data/SXS/SXS_BBH_"
        + N
        + "_Res"
        + str(round(simResLev[k]))
        + ".h5"
    )
    dT = 1.0 / (4096)
    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*fpeak_data[case_number]
    mm_list = []
    if mflag == 0:
        M = 37.5
    elif mflag == 1:
        M = 150
    elif mflag==2:
        M = 75
    elif mflag==3:
        M = 112.5
    elif mflag==4:
        M = 187.5
    elif mflag==5:
        M = 225
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phi in phi_vals:
        phic = phi
        hp_nr, hc_nr, pars = ReadSXSWithLAL(
            fgw, M, inclination, dT, distance, phic)
        m1 = pars["m1"]
        m2 = pars["m2"]
        c1x, c1y, c1z = chiA = pars["chiA"]
        c2x, c2y, c2z = chiB = pars["chiB"]
        f0 = pars["flow"]
        # if M==37.5:
        #     flow = f0+3
        # else:
        #     flow=11
        flow = f0+3
        pars_dict = EOB_par(M, m1/m2, f0, c1x, c1y, c1z, c2x,
                            c2y, c2z, distance, inclination)
        phic = phi
        pars_dict['coalescence_angle'] = phic
        hp_ts = TimeSeries(hp_nr.data.data, dT)
        hc_ts = TimeSeries(hc_nr.data.data, dT)

        t, hp_eob, hc_eob = EOBRun_module.EOBRunPy(pars_dict)
        hp_eobTS = TimeSeries(hp_eob, dT)
        hc_eobTS = TimeSeries(hc_eob, dT)
        tl = (max(len(hp_ts), len(hp_eobTS))-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hp_eobTS.resize(tlen)
        hc_eobTS.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()
        hpf_eob = hp_eobTS.to_frequencyseries()
        hcf_eob = hc_eobTS.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for kappa in kappa_vals:
            k = kappa
            h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr
            if (kappa == 0.0) and (phi == 0.0):
                m,  _ = match(hp_ts, hp_eobTS, psd=psd,
                              low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)

            def localfunc_spinpr(x):  # Anneal over approx :(
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # EOB wf
                t, hp_eob, hc_eob = EOBRun_module.EOBRunPy(pars_dict)
                hp_eobTS = TimeSeries(hp_eob, dT)
                hc_eobTS = TimeSeries(hc_eob, dT)
                hp_eobTS.resize(tlen)
                hc_eobTS.resize(tlen)
                hpf = hp_eobTS.to_frequencyseries()
                hcf = hc_eobTS.to_frequencyseries()
                ans = 1. - \
                    sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)
                return ans

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(Ni, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [Ni, 1-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]


def sxs_long_seob_mm(case_number, sxs_data, inclination, mr_flag, distance, maxFun, mflag, fpeak_data):
    data_number = case_number-1
    simNo = sxs_data[:, 0]
    simResLev = sxs_data[:, 3]
    k = case_number
    Ni = round(simNo[k])
    if Ni < 10:
        N = "000" + str(Ni)
    elif Ni < 100:
        N = "00" + str(Ni)
    elif Ni < 1000:
        N = "0" + str(Ni)
    else:
        N = str(Ni)

    # print("SXS:",N)
    fgw = (
        "data/SXS/SXS_BBH_"
        + N
        + "_Res"
        + str(round(simResLev[k]))
        + ".h5"
    )

    dT = 1.0 / (4096)
    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*fpeak_data[case_number]
    mm_list = []
    if mflag == 0:
        M = 37.5
    elif mflag == 1:
        M = 150
    elif mflag==2:
        M = 75
    elif mflag==3:
        M = 112.5
    elif mflag==4:
        M = 187.5
    elif mflag==5:
        M = 225
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phi in phi_vals:
        phic = phi
        hp_nr, hc_nr, pars = ReadSXSWithLAL(
            fgw, M, inclination, dT, distance, phic)
        m1 = pars["m1"]
        m2 = pars["m2"]
        c1x, c1y, c1z = chiA = pars["chiA"]
        c2x, c2y, c2z = chiB = pars["chiB"]
        f0 = pars["flow"]
        # if M==37.5:
        #     flow = f0+3
        # else:
        #     flow=11
        flow = f0+3
        pars_dict = EOB_par(M, m1/m2, f0, c1x, c1y, c1z, c2x,
                            c2y, c2z, distance, inclination)

        pars_dict['coalescence_angle'] = phic
        hp_ts = TimeSeries(hp_nr.data.data, dT)
        hc_ts = TimeSeries(hc_nr.data.data, dT)

        t, hp_eob, hc_eob = GenLALWfTD(pars_dict, approx='SEOBNRv4P')
        hp_eobTS = TimeSeries(hp_eob, dT)
        hc_eobTS = TimeSeries(hc_eob, dT)
        tl = (max(len(hp_ts), len(hp_eobTS))-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hp_eobTS.resize(tlen)
        hc_eobTS.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()
        hpf_eob = hp_eobTS.to_frequencyseries()
        hcf_eob = hc_eobTS.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for kappa in kappa_vals:
            k = kappa
            h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr
            if (kappa == 0.0) and (phi == 0.0):
                m,  _ = match(hp_ts, hp_eobTS, psd=psd,
                              low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)

            def localfunc_spinpr(x):  # Anneal over approx :(
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # EOB wf
                t, hp_eob, hc_eob = GenLALWfTD(pars_dict, approx='SEOBNRv4P')
                hp_eobTS = TimeSeries(hp_eob, dT)
                hc_eobTS = TimeSeries(hc_eob, dT)
                hp_eobTS.resize(tlen)
                hc_eobTS.resize(tlen)
                hpf = hp_eobTS.to_frequencyseries()
                hcf = hc_eobTS.to_frequencyseries()
                ans = 1. - \
                    sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)
                return ans

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(Ni, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [Ni, 1-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]


def sxs_long_seobV5_mm(case_number, sxs_data, inclination, mr_flag, distance, maxFun, mflag, fpeak_data):
    data_number = case_number-1
    simNo = sxs_data[:, 0]
    simResLev = sxs_data[:, 3]
    k = case_number
    Ni = round(simNo[k])
    if Ni < 10:
        N = "000" + str(Ni)
    elif Ni < 100:
        N = "00" + str(Ni)
    elif Ni < 1000:
        N = "0" + str(Ni)
    else:
        N = str(Ni)

    # print("SXS:",N)
    fgw = (
        "data/SXS/SXS_BBH_"
        + N
        + "_Res"
        + str(round(simResLev[k]))
        + ".h5"
    )

    dT = 1.0 / (4096)
    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*fpeak_data[case_number]
    mm_list = []
    if mflag == 0:
        M = 37.5
    elif mflag == 1:
        M = 150
    elif mflag==2:
        M = 75
    elif mflag==3:
        M = 112.5
    elif mflag==4:
        M = 187.5
    elif mflag==5:
        M = 225
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phi in phi_vals:
        phic = phi
        hp_nr, hc_nr, pars = ReadSXSWithLAL(
            fgw, M, inclination, dT, distance, phic)
        m1 = pars["m1"]
        m2 = pars["m2"]
        c1x, c1y, c1z = chiA = pars["chiA"]
        c2x, c2y, c2z = chiB = pars["chiB"]
        f0 = pars["flow"]
        # if M==37.5:
        #     flow = f0+3
        # else:
        #     flow=11
        flow=f0+3
        pars_dict = EOB_par(M, m1/m2, f0, c1x, c1y, c1z, c2x,
                            c2y, c2z, distance, inclination)

        pars_dict['coalescence_angle'] = phic
        hp_ts = TimeSeries(hp_nr.data.data, dT)
        hc_ts = TimeSeries(hc_nr.data.data, dT)

        t, hp_eob, hc_eob = GenSEOBV5(pars_dict, fhigh)
        hp_eobTS = TimeSeries(hp_eob.data.data, dT)
        hc_eobTS = TimeSeries(hc_eob.data.data, dT)
        tl = (max(len(hp_ts), len(hp_eobTS))-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hp_eobTS.resize(tlen)
        hc_eobTS.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()
        hpf_eob = hp_eobTS.to_frequencyseries()
        hcf_eob = hc_eobTS.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for kappa in kappa_vals:
            k = kappa
            h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr
            if (kappa == 0.0) and (phi == 0.0):
                m,  _ = match(hp_ts, hp_eobTS, psd=psd,
                              low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)

            def localfunc_spinpr(x):  # Anneal over approx :(
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # EOB wf
                t, hp_eob, hc_eob = GenSEOBV5(pars_dict, fhigh)
                hp_eobTS = TimeSeries(hp_eob.data.data, dT)
                hc_eobTS = TimeSeries(hc_eob.data.data, dT)
                hp_eobTS.resize(tlen)
                hc_eobTS.resize(tlen)
                hpf = hp_eobTS.to_frequencyseries()
                hcf = hc_eobTS.to_frequencyseries()
                ans = 1. - \
                    sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)
                return ans

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(Ni, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [Ni, 1-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]


def sxs_long_tphm_mm(case_number, sxs_data, inclination, mr_flag, distance, maxFun, mflag, fpeak_data):
    data_number = case_number-1
    simNo = sxs_data[:, 0]
    simResLev = sxs_data[:, 3]
    k = case_number
    Ni = round(simNo[k])
    if Ni < 10:
        N = "000" + str(Ni)
    elif Ni < 100:
        N = "00" + str(Ni)
    elif Ni < 1000:
        N = "0" + str(Ni)
    else:
        N = str(Ni)

    # print("SXS:",N)
    fgw = (
        "data/SXS/SXS_BBH_"
        + N
        + "_Res"
        + str(round(simResLev[k]))
        + ".h5"
    )

    dT = 1.0 / (4096)
    if mr_flag:
        fhigh = 1024
    else:
        fhigh = 0.6*fpeak_data[case_number]
    mm_list = []
    if mflag == 0:
        M = 37.5
    elif mflag == 1:
        M = 150
    elif mflag==2:
        M = 75
    elif mflag==3:
        M = 112.5
    elif mflag==4:
        M = 187.5
    elif mflag==5:
        M = 225
    phi_vals = np.array([0, 60, 120, 180, 240, 300])
    kappa_vals = np.array([0, 15, 30, 45, 60, 75, 90])
    phi_vals = (np.pi/180)*phi_vals
    kappa_vals = (np.pi/180)*kappa_vals
    for phi in phi_vals:
        phic = phi
        hp_nr, hc_nr, pars = ReadSXSWithLAL(
            fgw, M, inclination, dT, distance, phic)
        m1 = pars["m1"]
        m2 = pars["m2"]
        c1x, c1y, c1z = chiA = pars["chiA"]
        c2x, c2y, c2z = chiB = pars["chiB"]
        f0 = pars["flow"]
        # if M==37.5:
        #     flow = f0+3
        # else:
        #     flow=11
        flow = f0+3
        pars_dict = EOB_par(M, m1/m2, f0, c1x, c1y, c1z, c2x,
                            c2y, c2z, distance, inclination)
        phic = phi
        pars_dict['coalescence_angle'] = phic
        hp_ts = TimeSeries(hp_nr.data.data, dT)
        hc_ts = TimeSeries(hc_nr.data.data, dT)

        t, hp_eob, hc_eob = GenLALWfTD(pars_dict, approx="IMRPhenomTPHM")
        hp_eobTS = TimeSeries(hp_eob, dT)
        hc_eobTS = TimeSeries(hc_eob, dT)
        tl = (max(len(hp_ts), len(hp_eobTS))-1)*dT
        tN = nextpow2(tl)
        tlen = int(tN/dT) + 1
        hp_ts.resize(tlen)
        hc_ts.resize(tlen)
        hp_eobTS.resize(tlen)
        hc_eobTS.resize(tlen)
        hpf_nr = hp_ts.to_frequencyseries()
        hcf_nr = hc_ts.to_frequencyseries()
        hpf_eob = hp_eobTS.to_frequencyseries()
        hcf_eob = hc_eobTS.to_frequencyseries()
        delta_f = 1. / hp_ts.duration
        flen = tlen//2 + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        pars_dict['df'] = delta_f

        for kappa in kappa_vals:
            k = kappa
            h = np.cos(k) * hpf_nr + np.sin(k)*hcf_nr
            if (kappa == 0.0) and (phi == 0.0):
                m,  _ = match(hp_ts, hp_eobTS, psd=psd,
                              low_frequency_cutoff=flow, high_frequency_cutoff=fhigh)

            def localfunc_spinpr(x):  # Anneal over approx :(
                pars_dict['coalescence_angle'] = x[0]
                thetax = x[1]
                chiA_rot, chiB_rot = rotate_in_plane_spins(chiA, chiB, thetax)
                pars_dict['chi1x'], pars_dict['chi1y'], pars_dict['chi1z'] = chiA_rot
                pars_dict['chi2x'], pars_dict['chi2y'], pars_dict['chi2z'] = chiB_rot
                pars_dict['chi1'] = chiA_rot[2]
                pars_dict['chi2'] = chiB_rot[2]

                # EOB wf
                t, hp_eob, hc_eob = GenLALWfTD(
                    pars_dict, approx="IMRPhenomTPHM")
                hp_eobTS = TimeSeries(hp_eob, dT)
                hc_eobTS = TimeSeries(hc_eob, dT)
                hp_eobTS.resize(tlen)
                hc_eobTS.resize(tlen)
                hpf = hp_eobTS.to_frequencyseries()
                hcf = hc_eobTS.to_frequencyseries()
                ans = 1. - \
                    sky_and_time_maxed_match(h, hpf, hcf, psd, flow, fhigh)
                return ans

            opt_p, mm = dual_annealing(
                localfunc_spinpr, [(0., 2.*np.pi), (0., 2.*np.pi)], maxfun=maxFun)
            mm_list.append(mm)
    print(Ni, fhigh, 1-m, min(mm_list),
          stat.mean(mm_list), max(mm_list), stat.stdev(mm_list))
    return [Ni, 1-m, min(mm_list), stat.mean(mm_list), max(mm_list), stat.stdev(mm_list)]


def quantile(x, q, weights=None):
    #Taken from the corner package https://corner.readthedocs.io/en/latest/
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()
    
def chieff(q,chi1,theta1,chi2,theta2):
    return (1/(1+q))*(chi1*np.cos(theta1)+q*chi2*np.cos(theta2))

def chip(q,chi1,theta1,chi2,theta2):
    out=np.zeros(len(chi1))
    for i,(x,y) in enumerate(zip(chi1*np.sin(theta1),q*((4*q+3)/(4+3*q))*chi2*np.sin(theta2))):
        out[i]=np.max([x,y])
    return out
def chipInj(q,chi1,theta1,chi2,theta2):
    x=chi1*np.sin(theta1)
    y=q*((4*q+3)/(4+3*q))*chi2*np.sin(theta2)
    return np.max([x,y])
def chiperpD(q,chi1,theta1,chi2,theta2,phi1,phi2):
    m1=1/(1+q)
    m2=q/(1+q)
    s1mod=(m1**2)*chi1
    s2mod=(m2**2)*chi2
    return(s1mod*np.sin(theta1)+s2mod*np.sin(theta2))


def chiperpGen(q,chi1,theta1,chi2,theta2,phi1,phi2):
    omega=q*(4*q+3)/(4+3*q)
    t1=np.square(chi1*np.sin(theta1))
    t2=np.square(omega*chi2*np.sin(theta2))
    t3=2*omega*chi1*chi2*np.sin(theta1)*np.sin(theta2)*np.cos(phi2-phi1)
    return (np.sqrt(t1+t2+t3))


def chiperpJ(q,chi1,theta1,chi2,theta2,phi1,phi2):
    m1=1/(1+q)
    m2=q/(1+q)
    s1mod=(m1**2)*chi1
    s2mod=(m2**2)*chi2
    s1perp=np.array([s1mod*np.sin(theta1)*np.cos(phi1),s1mod*np.sin(theta1)*np.sin(phi1)])
    s2perp=np.array([s2mod*np.sin(theta2)*np.cos(phi2),s2mod*np.sin(theta2)*np.sin(phi2)])
    sperp=s1perp+s2perp
    return np.sqrt(sperp[0]*sperp[0]+sperp[1]*sperp[1])