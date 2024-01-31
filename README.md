# Survey_Precessing_Models

##
This repository contains three main directories: 
- data_files:
    - params: contains the intrinsic parameter files for the different data sets. E.g., discrete, uniform, short SXS etc.
    - pe_config_files: contains the config files for the parameter estimation runs.
- python: contains the python and Jupyter notebooks as well as shell scripts to run the jobs in large blocks. **Note:** file paths and other related objects need to be changed. Please read README.txt in the dir.
- results:
    - mismatches: contains the mismatch data for all the different data sets. See below for further details.
    - pe: contains the posterior results from the parameter estimation runs as .json files.
    - timing_results: contains the timing data for the three different sets of parameters used. See App. D of the [manuscript](https://arxiv.org/) for details. The files are of .json format again.

## Virtual Environments

Please see *Software* under the **Acknowledgements** section of the [manuscript](https://arxiv.org/) for software version information.  

## Structuring of the Mismatch Data

- The mismatch data contains the approximant strings: 'SEOB', 'TEOB', 'TPHM' and 'XPHM' representing the waveform model. 
- The strings 'NoMR', 'MR' pertain to whether the mismatch integral was truncated before the onset of the plunge (NoMr) or the integral was computed to 1024Hz (MR).
- The string 'inc0', 'inc90' indicate the inclination of the binary at the reference frequency.
- Each mismatch contains a header at row 0 describing the columns. The data column under the heading 'Mean[mismatch[fancy]]' is what is used the [manuscript](https://arxiv.org/) referred to as the sky-optimized mismatch.
