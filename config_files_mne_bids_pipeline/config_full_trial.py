study_name = "ABSeqMEG"

bids_root = "./Data/BIDS/"
deriv_root = "./Data/Data/mne-bids-pipeline/"

task = "abseq"
runs = 'all'

find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
mf_ctc_fname = bids_root + "/derivatives/meg_derivatives/ct_sparse_nspn.fif"
mf_cal_fname = bids_root + "/derivatives/meg_derivatives/sss_cal_nspn.dat"
ch_types = ["meg"]

l_freq = None
h_freq = 40.0

# here put the ICA thing

# Epochs
epochs_tmin = -0.2
epochs_tmax = 15
epochs_decim = 10
baseline = (None, 0)

# Conditions / events to consider when epoching
conditions = ["StimPosition_1"]

# Decoding
decode = False

# Noise estimation
process_empty_room = False
