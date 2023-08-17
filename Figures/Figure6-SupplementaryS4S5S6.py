'''
Figure 6. Sequence complexity in the proposed language of thought (LoT) modulates MEG signals to habituation, standard and deviant trials.

1 - Global field power analysis :
Global field power computed for each sequence from the evoked potentials of the Habituation, Standard and Deviant trials.
0 ms indicates sound onset. Note that the time-window ranges until 350 ms for Habituation and Standard trials (with a new sound onset at S0A=250 ms),
and until 600 ms for Deviant trials and for the others.
Significant correlation with sequence complexity was found in Habituation and Deviant GFPs and are indicated by the shaded areas.

2 - Regressions of MEG signals as a function of sequence complexity.
Left: amplitude of the regression coefficients ẞ of the complexity regressor for each MEG sensor. Insets show the projection
of those coefficients in source space at the maximal amplitude peak, indicated by a vertical dotted line.
Right: spatiotemporal clusters where regression coefficients were significantly different from 0.
'''

import config
import numpy as np
from functions import article_plotting_funcs, epoching_funcs, regression_funcs
import os.path as op

#  ----------------------------------------------- ----------------------------------------------- ---------------------
# ----------------------------------------------- GFP ANALYSIS ---------------------------------------------------------
#  ----------------------------------------------- ----------------------------------------------- ---------------------

def compute_GFP_asmag_allparticipants():
    gfp_data = {}
    for ttype in ['habituation', 'standard', 'violation']:
        gfp_data[ttype] = {}
        for seqID in range(1, 8):
            gfp_data[ttype][seqID] = []
    for subject in config.subjects_list:
        print('-- Subject -- \n' + subject)
        print(' -- LOAD THE EPOCHS and remap to only magnetometers -- #')
        epochs = epoching_funcs.load_epochs_items(subject, cleaned=True, AR_type='global')
        epochs = epochs.as_type('mag', mode="accurate")
        for ttype in ['habituation', 'standard', 'violation']:
            print('  ---- Trial type ' + str(ttype))
            for seqID in range(1, 8):
                print(' -- Computing GFP for sequence ' + str(seqID) + ' --\n')
                if ttype == 'habituation':
                    epochs_subset = epochs['TrialNumber <= 10 and SequenceID == ' + str(seqID)].copy()
                elif ttype == 'standard':  # For standards, taking only items from trials with no violation
                    epochs_subset = epochs[
                        'TrialNumber > 10 and SequenceID == ' + str(seqID) + ' and ViolationInSequence == 0'].copy()
                elif ttype == 'violation':
                    epochs_subset = epochs[
                        'TrialNumber > 10 and SequenceID == ' + str(seqID) + ' and ViolationOrNot == 1'].copy()
                ev = epochs_subset.average()
                gfp = np.sqrt(np.sum(ev.copy().pick_types(eeg=False, meg='mag').data ** 2, axis=0))
                gfp_data[ttype][seqID].append(gfp)
    gfp_data['times'] = ev.times
    return gfp_data


def plot_and_save_GFP_stats(gfp_data):
    f = open(op.join(config.fig_path, 'GFP', 'statistics.txt'), 'w')
    fnames = {'habituation': 'Habituation', 'standard': 'Standard', 'violation': 'Deviant'}
    for cond in fnames.keys():
        data_7seq = np.dstack(gfp_data[cond].values())
        data_7seq = np.transpose(data_7seq, (2, 0, 1))
        # Data line plot 7seq
        article_plotting_funcs.plot_7seq_timecourses(data_7seq, gfp_data['times'] * 1000, save_fig_path='GFP/',
                                                     fig_name='GFPxComplexity_' + fnames[cond], suffix='',
                                                     pos_horizontal_bar=0.47, plot_pearson_corrComplexity=True,
                                                     chance=None, xlims=[-50, 350], ymin=0, ylabel='GFP', logger=f)

        # Correlation with complexity heatmap
        print("---- we are determining the correlation with complexity for %s trials ")
        res = article_plotting_funcs.compute_corr_comp(data_7seq)
        ylims = [-0.5, 0.5]
        article_plotting_funcs.heatmap_avg_subj(res, gfp_data['times'] * 1000, xlims=[-50, 350],
                                                ylims=ylims, fig_name=op.join(config.fig_path, 'GFP',
                                                                              'GFPxComplexity_Habituation_heatmap_complexity' + 'pearson'),
                                                figsize=(10, 0.5), label='Beta')
    f.close()

gfp_data = compute_GFP_asmag_allparticipants()
plot_and_save_GFP_stats(gfp_data)

#  ----------------------------------------------- ----------------------------------------------- ---------------------
# ---------------------------------------- LINEAR REGRESSION ANALYSIS --------------------------------------------------
#  ----------------------------------------------- ----------------------------------------------- ---------------------

# 1 -  -  -  -  -  -  -  -  - -  -  -  -  -  -   Calculer les régressions -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
def linear_regressions(subject):
    print("--- APPENDING THE NEW PREDICTORS SURPRISE AND REPEAT/ALTERNATE and Complexity^2 INTO THE METADATA ---")
    regression_funcs.update_metadata_epochs_and_save_epochs(subject)
    filter_names = ['Hab', 'Stand', 'Viol']
    for filter_name in filter_names:
        print(" ----- ANALYSIS PRESENTED IN FIGURE 6B ------- ")
        regression_funcs.compute_regression(subject, ['Intercept', 'Complexity'], "", filter_name,
                                            remap_channels='grad_to_mag', apply_baseline=True, cleaned=True)
        print(" ----- ANALYSES PRESENTED IN FIGURE S4 S5 and S6 ------- ")
        regression_funcs.compute_regression(subject,
                                            ['Intercept', 'Complexity', 'surprise_100', 'Surprisenp1', 'RepeatAlter',
                                             'RepeatAlternp1'], "", filter_name, remap_channels='grad_to_mag',
                                            apply_baseline=True, cleaned=True)


# 2 -  -  -  -  -  -  -  -  -  - -  -  -  -  -  - Merge the per-subject results and plot the functions -  -  -  -  -  -
filter_names = ['Hab', 'Stand', 'Viol']
for filter_name in filter_names:
    print(" ----- ANALYSIS PRESENTED IN FIGURE 6B ------- ")
    regressors_names = ['Intercept', 'Complexity']
    regression_funcs.merge_individual_regression_results(regressors_names, "", filter_name,
                                                         suffix='--remapped_gtmbaselined')
    regression_funcs.regression_group_analysis(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined',
                                               Do3Dplot=True)

    print(" ----- ANALYSIS PRESENTED IN FIGURE S4 S5 and S6 ------- ")
    regressors_names = ['Intercept', 'Complexity', 'surprise_100', 'Surprisenp1', 'RepeatAlter', 'RepeatAlternp1']
    regression_funcs.merge_individual_regression_results(regressors_names, "", filter_name,
                                                         suffix='--remapped_gtmbaselined')
    regression_funcs.regression_group_analysis(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined',
                                               Do3Dplot=True)
