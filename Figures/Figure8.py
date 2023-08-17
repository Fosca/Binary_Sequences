"""
========================================================================================================================
Testing the Standard vs Deviant decoder on the full 16 item sequences - Full GAT matrix for each of the 7 sequences
========================================================================================================================
"""

# ---- import the packages -------
import config
import matplotlib.pyplot as plt
from functions import stats_funcs, SVM_funcs, utils
import numpy as np
import os.path as op
import mne
import matplotlib.ticker as ticker
from jr.plot import pretty_gat

#  ----------------------------------------------- ----------------------------------------------- ---------------------
#  ---------------------------------------- PLOT THE GAT PER SEQUENCES -------------------------------------------------
#  ----------------------------------------------- ----------------------------------------------- ---------------------

def results_SVM_standard_deviant(fname,subjects_list):
    """
    Function to load the results from the decoding of standard VS deviant
    """
    results = {sens: [] for sens in config.ch_types}
    times = []
    for sens in config.ch_types:
        results[sens] = {'SeqID_%i' % i: [] for i in range(1, 8)}
        results[sens]["average_all_sequences"] = []
        for subject in subjects_list:
            print("running the loop for subject %s \n"%subject)
            load_path = config.result_path+'/SVM/'+subject+'/'+fname
            data = np.load(load_path, allow_pickle=True).item()
            # Load the results
            data_GAT_sens = data['GAT'][sens]
            times = data['times']
            for seqID in range(1,8):
                results[sens]["SeqID_"+str(seqID)].append(data_GAT_sens["SeqID_"+str(seqID)])
            results[sens]["average_all_sequences"].append(data_GAT_sens["average_all_sequences"])
    return results, times

# ______________________________________________________________________________________________________________________
def plot_results_GAT_chans_seqID(results,times,save_folder,compute_significance=None,suffix='SW_train_different_blocks',chance = 0.5,clim=None):

    for chans in results.keys():
        res_chan = results[chans]
        for seqID in res_chan.keys():
            res_chan_seq = np.asarray(res_chan[seqID])
            sig_all = None
            # ---- compute significance ----
            if compute_significance is not None:
                tmin_sig = compute_significance[0]
                tmax_sig = compute_significance[1]
                times_sig = np.where(np.logical_and(times <= tmax_sig, times > tmin_sig))[0]
                sig_all = np.ones(res_chan_seq[0].shape)
                GAT_all_for_sig = res_chan_seq[:, times_sig, :]
                GAT_all_for_sig = GAT_all_for_sig[:, :, times_sig]
                sig = stats_funcs.stats(GAT_all_for_sig - chance, tail=1)
                sig_all = SVM_funcs.replace_submatrix(sig_all, times_sig, times_sig, sig)
            # -------- plot the gat --------
            ax = pretty_gat(np.nanmean(res_chan_seq,axis=0),times=times,sig=sig_all<0.05,chance = 0.5,clim=clim)
            plt.gcf().savefig(config.fig_path+save_folder+'/'+chans+'_'+seqID+suffix+'.png')
            plt.gcf().savefig(config.fig_path+save_folder+'/'+chans+'_'+seqID+suffix+'.svg')
            plt.close('all')

results, times = results_SVM_standard_deviant('SW_train_different_blocks_cleanedGAT_results.npy',config.subjects_list)
plot_results_GAT_chans_seqID(results,times,'/SVM/standard_vs_deviant/GAT/',compute_significance=[0,0.6],suffix='_cleaned_SW',clim=[0.37,0.63])

#  ----------------------------------------------- ----------------------------------------------- ---------------------
#  ------------------------------------------- DECODERS PROJECTION ON THE 16 ITEM SEQUENCES ----------------------------
#  ----------------------------------------------- ----------------------------------------------- ---------------------

# 1 - COMPUTE THE PROJECTION OF THE DECODERS FROM THE TIME-WINDOW : We take 131 ms - 210 ms because when we average the decoder
# performance on the diagonal, this window has the highest decoding score.

def apply_decoder_on_full_sequences(subject):
    SVM_funcs.generate_SVM_all_sequences(subject)
    SVM_funcs.apply_SVM_filter_16_items_epochs(subject, times=[0.131, 0.210], sliding_window=True)
    SVM_funcs.apply_SVM_filter_16_items_epochs_habituation(subject, times=[0.131, 0.210], sliding_window=True)


for subject in config.subjects_list:
    apply_decoder_on_full_sequences(subject)

# 2 - Load and reshape the projections that were saved as epochs -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

epochs_16_hab = []
epochs_16_test = []

subjects_list = ['sub01-pa_190002', 'sub02-ch_180036', 'sub03-mr_190273', 'sub04-rf_190499', 'sub05-cr_170417', 'sub06-kc_160388',
                 'sub07-jm_100109', 'sub08-cc_150418', 'sub09-ag_170045', 'sub10-gp_190568', 'sub11-fr_190151', 'sub12-lg_170436',
                 'sub13-lq_180242', 'sub14-js_180232', 'sub15-ev_070110', 'sub16-ma_190185', 'sub17-mt_170249', 'sub18-eo_190576',
                 'sub19-mg_190180']

for subject in subjects_list:
    epochs_16_test.append(mne.read_epochs(op.join(config.meg_dir, subject, 'as_magSW__131_210ms_SVM_on_16_items_test_window-epo.fif')))
    epochs_16_hab.append(mne.read_epochs(op.join(config.meg_dir, subject, 'as_magSW__131_210ms_SVM_on_16_items_habituation_window-epo.fif')))

# 3 and now plot all the projections -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

save_folder = op.join(config.fig_path, 'SVM', 'Full_sequence_projection')
utils.create_folder(save_folder)

epochs_list = {}
win_tmin = 131
win_tmax = 210

epochs_16 = {'hab':epochs_16_hab,'test':epochs_16_test}

plot_SVM_projection_for_seqID_window_allseq_heatmap(epochs_16,  window_CBPT_violation = 0.6,sensor_type='as_mag',
                                                              save_path=op.join(save_folder, 'AllSeq_%s_window_%i_%ims_final_no_interpolation_16.svg' % ('as_mag', win_tmin, win_tmax)),
                                                              vmin=-1,vmax=1,font_size=16)

plot_SVM_projection_for_seqID_window_allseq_heatmap(epochs_16,  window_CBPT_violation = 0.6,sensor_type='as_mag',
                                                              save_path=op.join(save_folder, 'AllSeq_%s_window_%i_%ims_final_no_interpolation_12.svg' % ('as_mag', win_tmin, win_tmax)),
                                                              vmin=-1,vmax=1,font_size=12)


plot_SVM_projection_for_seqID_window_allseq_heatmap(epochs_16,  window_CBPT_violation = 0.6,sensor_type='as_mag',
                                                              save_path=op.join(save_folder, 'AllSeq_%s_window_%i_%ims_final_no_interpolation_18.svg' % ('as_mag', win_tmin, win_tmax)),
                                                              vmin=-1,vmax=1,font_size=18)
# ______________________________________________________________________________________________________________________
def plot_SVM_projection_for_seqID_window_allseq_heatmap(epochs_list, sensor_type, save_path=None, vmin=-1, vmax=1,window_CBPT_violation = None,font_size = 16):

    color_viol = ['lightgreen','mediumseagreen','mediumslateblue','darkviolet']

    # window info, just for figure title
    win_tmin = epochs_list['test'][0][0].metadata.SVM_filter_tmin_window[0] * 1000
    win_tmax = epochs_list['test'][0][0].metadata.SVM_filter_tmax_window[0] * 1000
    n_plots = 7


    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 12), sharex=True, sharey=False, constrained_layout=True)
    # fig.suptitle('SVM %s - window %d-%dms; N subjects = %d' % (
    #     sensor_type, win_tmin, win_tmax, len(epochs_list['test'])), fontsize=12)

    ax = axes.ravel()[::1]
    ax[0].set_title('Repeat\n', loc='left', weight='bold',fontsize = font_size)
    ax[1].set_title('Alternate\n', loc='left', weight='bold',fontsize = font_size)
    ax[2].set_title('Pairs\n', loc='left', weight='bold',fontsize = font_size)
    ax[3].set_title('Quadruplets\n', loc='left', weight='bold',fontsize = font_size)
    ax[4].set_title('Pairs&Alt1\n', loc='left', weight='bold',fontsize = font_size)
    ax[5].set_title('Shrinking\n', loc='left', weight='bold',fontsize = font_size)
    ax[6].set_title('Complex\n', loc='left', weight='bold',fontsize = font_size)

    seqtxtXY = ['AAAAAAAAAAAAAAAA',
                'ABABABABABABABAB',
                'AABBAABBAABBAABB',
                'AAAABBBBAAAABBBB',
                'AABBABABAABBABAB',
                'AAAABBBBAABBABAB',
                'ABAAABBBBABBAAAB']

    print("vmin = %0.02f, vmax = %0.02f" % (vmin, vmax))

    n = 0

    violation_significance = {i:[] for i in range(1, 8)}
    epochs_data_hab_allseq = []
    epochs_data_test_allseq = []

    for seqID in range(1, 8):
        print("=== running for sequence %i ==="%seqID)
        #  this provides us with the position of the violations and the times
        epochs_seq_subset = epochs_list['test'][0]['SequenceID == ' + str(seqID) ]
        times = epochs_seq_subset.times
        times = times + 0.3
        violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])
        violation_significance[seqID] = {'times':times,'window_significance':window_CBPT_violation}

        #  ----------- habituation trials -----------
        epochs_data_hab_seq = []
        y_list_epochs_hab = []
        data_nanmean = []
        where_sig = []
        for epochs in epochs_list['hab']:
            epochs_subset = epochs['SequenceID == ' + str(seqID) ]
            avg_epo = np.nanmean(np.squeeze(epochs_subset.get_data()), axis=0)
            y_list_epochs_hab.append(avg_epo)
            epochs_data_hab_seq.append(avg_epo)
        epochs_data_hab_allseq.append(epochs_data_hab_seq)
        nanmean_hab = np.nanmean(y_list_epochs_hab, axis=0)
        data_nanmean.append(nanmean_hab)
        where_sig.append(np.zeros(nanmean_hab.shape))
        where_sig.append(np.zeros(nanmean_hab.shape))

        #  ----------- test trials -----------
        epochs_data_test_seq = []

        for viol_pos in violpos_list:
            y_list = []
            contrast_viol_pos = []
            for epochs in epochs_list['test']:
                epochs_subset = epochs[
                    'SequenceID == ' + str(seqID) + ' and ViolationInSequence == ' + str(viol_pos) ]
                avg_epo = np.nanmean(np.squeeze(epochs_subset.get_data()), axis=0)
                y_list.append(avg_epo)
                if viol_pos==0:
                    avg_epo_standard = np.nanmean(np.squeeze(epochs_subset.get_data()), axis=0)
                    epochs_data_test_seq.append(avg_epo_standard)
                if viol_pos !=0 and window_CBPT_violation is not None:
                    epochs_standard = epochs[
                        'SequenceID == ' + str(seqID) + ' and ViolationInSequence == 0']
                    avg_epo_standard = np.nanmean(np.squeeze(epochs_standard.get_data()), axis=0)
                    contrast_viol_pos.append(avg_epo - avg_epo_standard)

            # --------------- CBPT to test for significance ---------------
            if window_CBPT_violation is not None and viol_pos !=0:
                time_start_viol = 0.250 * (viol_pos - 1)
                time_stop_viol = time_start_viol + window_CBPT_violation
                inds_stats = np.where(np.logical_and(times>time_start_viol,times<=time_stop_viol))
                contrast_viol_pos = np.asarray(contrast_viol_pos)
                p_vals = np.asarray([1]*contrast_viol_pos.shape[1])
                p_values = stats_funcs.stats(contrast_viol_pos[:, inds_stats[0]], tail=1)
                p_vals[inds_stats[0]] = p_values
                violation_significance[seqID][int(viol_pos)] = p_vals
                y_list_alpha = 1*(p_vals<0.05)
                where_sig.append(y_list_alpha)

            nanmean_y = np.nanmean(y_list, axis=0)
            data_nanmean.append(nanmean_y)
        epochs_data_test_allseq.append(epochs_data_test_seq)
        where_sig = np.asarray(where_sig)

        width = 75
        # Add vertical lines, and "xY"
        for xx in range(16):
            ax[n].axvline(250 * xx,ymin=0,ymax= width, linestyle='--', color='black', linewidth=0.8)
            txt = seqtxtXY[n][xx]
            ax[n].text(250 * (xx + 1) - 125, width * 6 + (width / 3), txt, horizontalalignment='center', fontsize=12)

        # return data_nanmean
        ax[n].spines["top"].set_visible(False)
        ax[n].spines["right"].set_visible(False)
        ax[n].spines["bottom"].set_visible(False)
        ax[n].spines["left"].set_visible(False)

        im = ax[n].imshow(data_nanmean, extent=[min(times) * 1000, max(times) * 1000, 0, 6 * width], cmap='RdBu_r',
                          vmin=vmin, vmax=vmax, interpolation='nearest')
        # add colorbar
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        # cb = fig.colorbar(im, ax=ax[n], location='right', format=fmt, shrink=.50, aspect=10, pad=.005)
        # cb.ax.yaxis.set_offset_position('left')
        # cb.set_label('a. u.')
        ax[n].set_yticks(np.arange(width / 2, 6 * width, width))
        ax[n].set_yticklabels(['Deviant - %d' % violpos_list[4], 'Deviant - %d' % violpos_list[3],
                               'Deviant - %d' % violpos_list[2], 'Deviant - %d' % violpos_list[1],
                               'Standard', 'Habituation'],fontsize=10)
        ax[n].axvline(0, linestyle='-', color='black', linewidth=2)

        # add deviant marks
        for k in range(4):
            viol_pos = violpos_list[k + 1]
            x = 250 * (viol_pos - 1)
            y1 = (4 - k) * width
            y2 = (4 - 1 - k) * width
            ax[n].plot([x, x], [y1, y2], linestyle='-', color='black', linewidth=6)
            ax[n].plot([x, x], [y1, y2], linestyle='-', color=color_viol[k], linewidth=3)

            find_where_sig = np.where(where_sig[k+2,:]==1)[0]
            if len(find_where_sig)!=0:
                ax[n].plot([1000 * times[find_where_sig[0]], 1000 * times[find_where_sig[-1]]], [-(k+1)*width/3, -(k+1)*width/3], linestyle='-', color=color_viol[k], linewidth=3)
        n += 1

    axes.ravel()[-1].set_xlabel('Time (ms)')
    fig.subplots_adjust(hspace=0.6)
    figure = plt.gcf()
    if save_path is not None:
        figure.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')

    return figure

