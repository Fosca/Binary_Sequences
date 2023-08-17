"""
===================================================================================================================================
Decoding of the Standard VS Deviants with SVM per sequence and per condition (Habituation, Standard, Deviant and Standard - Deviant
===================================================================================================================================
# DESCRIPTION OF THE ANALYSIS
# We trained a decoder on all the sequences on half of the data (1 out of the 2 runs) to decode if a sound was standard
# or deviant. The selected standard training trials were matched with deviants ordinal positions in the sequence. We then tested
# these decoders on the Habituation trials and the Standard - Deviant trials.
# The selected data was cleaned with autoreject global
"""
import config
from functions import article_plotting_funcs, SVM_funcs
import os.path as op
import numpy as np
import matplotlib

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

#  ----------------------------------------------- ----------------------------------------------- ---------------------
# ------------------------------- RUN PER SUBJECT THE DECODING ANALYSIS ------------------------------------------------
#  ----------------------------------------------- ----------------------------------------------- ---------------------

subjects_list = config.subjects_list
for subject in subjects_list:
    SVM_funcs.generate_SVM_all_sequences(subject)
    SVM_funcs.GAT_SVM_trained_all_sequences(subject, metric='projection_normal')
    SVM_funcs.GAT_SVM_trained_all_sequences(subject, metric='score')

#  ----------------------------------------------- ----------------------------------------------- ---------------------
#  ------------------------------------- PLOT THE DECODING RESULTS -----------------------------------------------------
#  ----------------------------------------------- ----------------------------------------------- ---------------------

sens = 'as_mag'
n_subjects = len(config.subjects_list)
name = "SW_train_different_blocks_cleanedGAT_results"
conditions = ["_hab", ""]


# 1 - load all the participants results and reshape
for suffix in conditions:
    filename = name + suffix
    results = {'SeqID_%i' % i: [] for i in range(1, 8)}
    n_subj = 0
    for subject in subjects_list:
        GAT_path = op.join(config.SVM_path, subject, filename + '.npy')
        print("---- loading data for subject %s ----- " % subject)
        if op.exists(GAT_path):
            GAT_results = np.load(GAT_path, allow_pickle=True).item()
            times = 1000 * GAT_results['times']
            GAT_results = GAT_results['GAT']
            for key in ['SeqID_%i' % i for i in range(1, 8)]:
                results[key].append(GAT_results[sens][key])
            n_subj += 1
        else:
            print("Missing data for %s " % GAT_path)
    reshaped_data = np.zeros((7, n_subj, len(times)))

# 2 - select the diagonal of the GAT, compute the pearson correlation, plot
for suffix in conditions:
    for ii, SeqID in enumerate(range(1, 8)):
        perform_seqID = np.asarray(results['SeqID_' + str(SeqID)])
        diago_seq = np.diagonal(perform_seqID, axis1=1, axis2=2)
        reshaped_data[ii, :, :] = diago_seq
    article_plotting_funcs.plot_7seq_timecourses(reshaped_data, times, save_fig_path='SVM/standard_vs_deviant/',
                                                 fig_name='All_sequences_standard_VS_deviant_cleaned_',
                                                 suffix=suffix + sens,
                                                 pos_horizontal_bar=0.47, plot_pearson_corrComplexity=True, chance=0.5)
    pearson = article_plotting_funcs.compute_corr_comp(reshaped_data)
    article_plotting_funcs.heatmap_avg_subj(pearson, times, xlims=None, ylims=[-.5, .5], filter=False,
                                            fig_name=config.fig_path + '/SVM/standard_vs_deviant/heatmap_complexity_pearson_'
                                                     + suffix + sens + '.svg', figsize=(10 * 0.8, 1))
