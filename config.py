"""
===========
Config file
===========
"""
import os
from collections import defaultdict
from sys import platform
import sys

# - - - - - - - - - - - - - - - -  sequence complexities - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
complexity = {1: 4, 2: 6, 3: 6, 4: 6, 5: 12, 6: 15, 7: 28}

# - - - - - - - - - - - - - - - -   colormaps we chose   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
seqcolors = [[0.2943, 0.0101, 0.6298], [0.4905, 0.0106, 0.6584], [0.6622, 0.1358, 0.5877], [0.7964, 0.278, 0.4713],
             [0.9007, 0.4233, 0.3609], [0.973, 0.5844, 0.2524], [0.9931, 0.7709, 0.1551]]

# - - - - - - - - - - - - - - - - channel types we consider - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
ch_types = ['mag', 'grad', 'all_chans', 'as_mag']

# - - - - - - - - - - - - - - - - setting up the directories - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if os.name == 'nt':
    root_path = 'Z:' + os.path.sep
elif os.name == 'posix':
    if platform == "linux" or platform == "linux2":
        root_path = '/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/'
    elif platform == "darwin":
        root_path = '//Volumes/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/'

study_path = os.path.join(root_path, 'data') + os.path.sep
result_path = os.path.join(root_path, 'results_PUBLICATION') + os.path.sep
cluster_path = os.path.join(root_path, 'scripts', 'ABSeq_PUBLICATION', 'cluster') + os.path.sep
scripts_path = os.path.join(root_path, 'scripts', 'ABSeq_PUBLICATION') + os.path.sep
BIDS_path = os.path.join(study_path, 'BIDS') + os.path.sep
SVM_path = os.path.join(result_path, 'SVM') + os.path.sep
decoding_path = os.path.join(result_path, 'decoding') + os.path.sep
GFP_path = os.path.join(result_path, 'GFP') + os.path.sep
linear_models_path = os.path.join(result_path, 'linear_models') + os.path.sep

sys.path.append(root_path)
sys.path.append(scripts_path)

# ``subjects_dir`` : str
#   The ``subjects_dir`` contains the MRI files for all subjects.

subjects_dir = os.path.join(study_path, 'subjects')
fig_path = os.path.join(root_path, 'figures_PUBLICATION')
meg_dir = os.path.join(study_path, 'MEG')
run_info_dir = os.path.join(study_path, 'run_info')

# - - - - - - - - - - - - - - - - study, subjects and runs  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
study_name = 'ABseq'
subjects_list = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
                 'sub-11',
                 'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19']

runs = ['run01', 'run02', 'run03', 'run04', 'run05', 'run06', 'run07',
        'run08', 'run09', 'run10', 'run11', 'run12', 'run13', 'run14']

runs_dict = {subject: runs for subject in subjects_list}

runs_dict['sub-03'] = ['run01', 'run02', 'run03', 'run04', 'run05', 'run06', 'run07',
                       'run08', 'run09', 'run10', 'run11',
                       'run12']  # importation error from MEG machine (unreadable files)
runs_dict['sub-07'] = ['run01', 'run02', 'run03', 'run04', 'run06', 'run07',
                       'run08', 'run09', 'run10', 'run11', 'run12', 'run13',
                       'run14']  # skipped a run during acquisition


# - - - - - - - - - - - - - - - - bad channels were identified visually  - - - - - - - - - - - - - - - - - - - - - - - -
def default_bads():
    return {name: [] for name in runs}


bads = defaultdict(default_bads)

bads['sub-01'] = {'run01': ['MEG0213', 'MEG1323', 'MEG2233', 'MEG1732'],
                  'run02': ['MEG0213', 'MEG1323', 'MEG2233', 'MEG1732'],
                  'run03': ['MEG0213', 'MEG1323', 'MEG2233', 'MEG1732'],
                  'run04': ['MEG0213', 'MEG1323', 'MEG2233', 'MEG1732', 'MEG0741'],
                  'run05': ['MEG0213', 'MEG1732'],
                  'run06': ['MEG0213', 'MEG1732'],
                  'run07': ['MEG0213', 'MEG1732'],
                  'run08': ['MEG0213', 'MEG0242'],
                  'run09': ['MEG0213', 'MEG1732'],
                  'run10': ['MEG0213', 'MEG1732', 'MEG2132'],
                  'run11': ['MEG0213', 'MEG1732'],
                  'run12': ['MEG0213', 'MEG1732'],
                  'run13': ['MEG0213', 'MEG1732'],
                  'run14': ['MEG0213', 'MEG1732', 'MEG1332', 'MEG0242']}

bads['sub-02'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321', 'MEG0632'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321'],
                  'run05': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321', 'MEG1241'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321', 'MEG0632'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321', 'MEG0632'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321', 'MEG2113'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2321', 'MEG1241'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run13': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run14': ['MEG0213', 'MEG0311', 'MEG2643']}

bads['sub-03'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'MEG0541', 'MEG0131'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'MEG1831'],
                  'run05': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'MEG1831'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'MEG1831']}

bads['sub-04'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2243'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2243'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2241'],
                  'run05': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1943', 'MEG1941'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1943', 'MEG2241'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1443', 'MEG1941'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1443', 'MEG1942'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1443', 'MEG1941'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1941'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2321'],
                  'run13': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1443', 'MEG1942', 'MEG2311'],
                  'run14': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1443', 'MEG1942', 'MEG1941']}

bads['sub-05'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131', 'MEG0522'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131'],
                  'run05': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run13': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131'],
                  'run14': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131']}

bads['sub-06'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1831', 'MEG2221', 'MEG0313', 'MEG2313'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1831', 'MEG2221', 'MEG0313', 'MEG2313'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0132'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run05': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121', 'MEG2113'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0111', 'MEG0321'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121', 'MEG0321', 'MEG0111'],
                  'run13': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121', 'MEG0321', 'MEG0111'],
                  'run14': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121', 'MEG0321', 'MEG0111']}

bads['sub-07'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231'],
                  'run13': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231'],
                  'run14': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231']}

bads['sub-08'] = {
    'run01': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2221', 'MEG1821', 'MEG1522', 'MEG1822', 'MEG1813''MEG2222',
              'MEG2223'],
    'run02': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522'],
    'run03': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522'],
    'run04': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522'],
    'run05': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522'],
    'run06': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522'],
    'run07': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522'],
    'run08': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522'],
    'run09': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522'],
    'run10': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522'],
    'run11': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522'],
    'run12': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'MEG1823', 'MEG0121'],
    'run13': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'MEG1823', 'MEG0121'],
    'run14': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'MEG1823', 'MEG0121']}
# For this participant, we had some problems when concatenating the raws for run08. The error message said that raw08._cals didn't match the other ones.
# We saw that it is the 'calibration' for the channel EOG061 that was different with respect to run09._cals.
# np.where(raw_list[7]._cals-raw_list[8]._cals)
# raw_list[7].info['ch_names'][382]
# We replaced by hand run08._cals by run09._cals and saved it.
# raw_list[7]._cals = raw_list[8]._cals

bads['sub-09'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG0813', 'MEG2642', 'MEG1731'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG1731'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG1731'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733'],
                  'run05': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG1731'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG1731'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG1731'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG1731'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG2642'],
                  'run13': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733'],
                  'run14': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733']}

bads['sub-10'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2412', 'MEG0522', 'MEG2641', 'MEG1111'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2412', 'MEG0522', 'MEG1111'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2412', 'MEG1111'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2412', 'MEG0522'],
                  'run05': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2323', 'MEG2422'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2412', 'MEG2242', 'MEG0522'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1333', 'MEG1332', 'MEG0522'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2423', 'MEG1612', 'MEG0122'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1612', 'MEG0522'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2412', 'MEG0522'],
                  'run13': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2012', 'MEG0522'],
                  'run14': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2423', 'MEG0522']}

bads['sub-11'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522'],
                  'run05': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG1831', 'MEG0121', 'MEG1841'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2423', 'MEG2422'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2422'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG1841', 'MEG2421', 'MEG2422'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2421', 'MEG2422'],
                  'run13': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2422'],
                  'run14': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2422', 'MEG1931']}

bads['sub-12'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG1523'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG2113', 'MEG1831'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG1523'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG0522'],
                  'run05': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG2113'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG1523'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG1523', 'MEG1613'],
                  'run13': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522'],
                  'run14': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG1523', 'MEG1613']}

bads['sub-13'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run05': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2221', 'MEG0141', 'MEG1221'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0141'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0141', 'MEG1521'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1541', 'MEG1221'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1221'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run13': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1221', 'MEG1442', 'MEG0613'],
                  'run14': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121', 'MEG1442']}

bads['sub-14'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1222', 'MEG2511'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1222', 'MEG2511'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run05': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1222'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run13': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run14': ['MEG0213', 'MEG0311', 'MEG2643']}

bads['sub-15'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0341', 'MEG2211', 'MEG2512'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2211', 'MEG2512'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2211', 'MEG2512', 'MEG2511'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2211', 'MEG0521', 'MEG0522', 'MEG0523', 'MEG2512',
                            'MEG2642'],
                  'run05': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2641', 'MEG2641', 'MEG2642', 'MEG2523', 'MEG2522'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2512', 'MEG2511'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2512'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2512'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2512'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2512'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2512', 'MEG1122'],
                  'run13': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2323', 'MEG0522', 'MEG1721'],
                  'run14': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2323', 'MEG0523', 'MEG1721']}

bads['sub-16'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run05': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0332'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2241', 'MEG0541', 'MEG0821'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0821'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0813', 'MEG0321'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0813'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0813'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run13': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run14': ['MEG0213', 'MEG0311', 'MEG2643']}

bads['sub-17'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG0513'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2642'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2642', 'MEG0813'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG2642', 'MEG0443', 'MEG1731'],
                  'run05': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1641'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run13': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run14': ['MEG0213', 'MEG0311', 'MEG2643']}

bads['sub-18'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141'])

bads['sub-19'] = {'run01': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run02': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run03': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run04': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run05': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run06': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run07': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run08': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run09': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run10': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run11': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run12': ['MEG0213', 'MEG0311', 'MEG2643', 'MEG1133'],
                  'run13': ['MEG0213', 'MEG0311', 'MEG2643'],
                  'run14': ['MEG0213', 'MEG0311', 'MEG2643']}