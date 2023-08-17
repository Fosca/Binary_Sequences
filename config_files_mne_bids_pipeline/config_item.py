import mne
import numpy as np

study_name = "ReplaySeq"
bids_root = "/Users/fosca/Documents/Fosca/INSERM/Projets/ReplaySeq/Data/BIDS/"
deriv_root = "/Users/fosca/Documents/Fosca/INSERM/Projets/ReplaySeq/Data/Data/mne-bids-pipeline/"

task = "reproduction"

runs = 'all'

find_flat_channels_meg = True
find_noisy_channels_meg = True
use_maxwell_filter = True
mf_ctc_fname = bids_root + "/derivatives/meg_derivatives/ct_sparse_nspn.fif"
mf_cal_fname = bids_root + "/derivatives/meg_derivatives/sss_cal_nspn.dat"
ch_types = ["meg"]

l_freq = None
h_freq = 40.0

# SSP and peak-to-peak rejection
spatial_filter = "ssp"
n_proj_eog = dict(n_mag=0, n_grad=0)
n_proj_ecg = dict(n_mag=2, n_grad=2)
reject = ssp_reject_ecg = {"grad": 2000e-13, "mag": 5000e-15}

# Epochs
epochs_tmin = -0.2
epochs_tmax = 0.6
epochs_decim = 4
baseline = None

# Conditions / events to consider when epoching
conditions = ['SequenceID-Rep2/Position-1', 'SequenceID-Rep2/Position-2', 'SequenceID-Rep2/Position-3', 'SequenceID-Rep2/Position-4', 'SequenceID-Rep2/Position-5', 'SequenceID-Rep2/Position-6', 'SequenceID-CRep2/Position-1', 'SequenceID-CRep2/Position-2', 'SequenceID-CRep2/Position-3', 'SequenceID-CRep2/Position-4', 'SequenceID-CRep2/Position-5', 'SequenceID-CRep2/Position-6', 'SequenceID-Rep3/Position-1', 'SequenceID-Rep3/Position-2', 'SequenceID-Rep3/Position-3', 'SequenceID-Rep3/Position-4', 'SequenceID-Rep3/Position-5', 'SequenceID-Rep3/Position-6', 'SequenceID-CRep3/Position-1', 'SequenceID-CRep3/Position-2', 'SequenceID-CRep3/Position-3', 'SequenceID-CRep3/Position-4', 'SequenceID-CRep3/Position-5', 'SequenceID-CRep3/Position-6', 'SequenceID-Rep4/Position-1', 'SequenceID-Rep4/Position-2', 'SequenceID-Rep4/Position-3', 'SequenceID-Rep4/Position-4', 'SequenceID-Rep4/Position-5', 'SequenceID-Rep4/Position-6', 'SequenceID-CRep4/Position-1', 'SequenceID-CRep4/Position-2', 'SequenceID-CRep4/Position-3', 'SequenceID-CRep4/Position-4', 'SequenceID-CRep4/Position-5', 'SequenceID-CRep4/Position-6', 'SequenceID-RepEmbed/Position-1', 'SequenceID-RepEmbed/Position-2', 'SequenceID-RepEmbed/Position-3', 'SequenceID-RepEmbed/Position-4', 'SequenceID-RepEmbed/Position-5', 'SequenceID-RepEmbed/Position-6', 'SequenceID-C1RepEmbed/Position-1', 'SequenceID-C1RepEmbed/Position-2', 'SequenceID-C1RepEmbed/Position-3', 'SequenceID-C1RepEmbed/Position-4', 'SequenceID-C1RepEmbed/Position-5', 'SequenceID-C1RepEmbed/Position-6', 'SequenceID-C2RepEmbed/Position-1', 'SequenceID-C2RepEmbed/Position-2', 'SequenceID-C2RepEmbed/Position-3', 'SequenceID-C2RepEmbed/Position-4', 'SequenceID-C2RepEmbed/Position-5', 'SequenceID-C2RepEmbed/Position-6']


# Decoding
decode = False

# Noise estimation

process_empty_room = False


# noise_cov == None