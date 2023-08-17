# This module contains all the functions related to the decoding analysis
import sys
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_PUBLICATION/')
import os.path as op

import pandas as pd
import mne
import time
import config
from functions import utils, stats_funcs

from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

from sklearn.base import TransformerMixin
from functions import epoching_funcs

# ______________________________________________________________________________________________________________________
def SVM_decoder():
    """
    Builds an SVM decoder that will be able to output the distance to the hyperplane once trained on data.
    It is meant to generalize across time by construction.
    :return:
    """
    clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=True))
    time_gen = GeneralizingEstimator(clf, scoring=None, n_jobs=8, verbose=True)

    return time_gen

# ----------------------------------------------------------------------------------------------------------------------
def train_test_different_blocks(epochs,return_per_seq = False):

    import random

    """
    For each sequence, check that there are two run_numbers for the sequence.
    If there are 2, then randomly put one in the training set and the other one in the test set.
    If there is just one, split the trials of that one into two sets, one for training the other for testing
    :param epochs:
    :return:
    """

    train_test_dict = {i:{'train':[],'test':[]} for i in range(1,8)}
    train_inds_fold1 = []
    train_inds_fold2 = []
    test_inds_fold1 = []
    test_inds_fold2 = []

    for seqID in range(1,8):
        epochs_Seq = epochs["SequenceID == %i "%seqID]
        n_runs = np.unique(epochs_Seq.metadata['RunNumber'].values)
        if len(n_runs) == 1:
            print('There is only one run for sequence ID %i'%(seqID))
            inds_seq = np.where(epochs.metadata['RunNumber'].values==n_runs)[0]
            np.random.shuffle(inds_seq)
            inds_1 = inds_seq[:int(np.floor(len(inds_seq)/2))]
            inds_2 = inds_seq[int(np.floor(len(inds_seq)/2)):]
        else:
            pick_run = random.randint(0, 1)
            run_train = n_runs[pick_run]
            run_test = n_runs[1-pick_run]
            inds_1 = np.where(epochs.metadata['RunNumber'].values==run_train)[0]
            inds_2 = np.where(epochs.metadata['RunNumber'].values==run_test)[0]

        train_inds_fold1.append(inds_1)
        train_inds_fold2.append(inds_2)

        test_inds_fold1.append(inds_2)
        test_inds_fold2.append(inds_1)

        train_test_dict[seqID]['train']= [inds_1,inds_2]
        train_test_dict[seqID]['test']= [inds_2,inds_1]

    if return_per_seq:
        return train_test_dict
    else:
        return [np.concatenate(train_inds_fold1),np.concatenate(train_inds_fold2)], [np.concatenate(test_inds_fold1),np.concatenate(test_inds_fold2)]

# ______________________________________________________________________________________________________________________
def generate_SVM_all_sequences(subject, sliding_window=True,cleaned = True):
    """
    Generates the SVM decoders for all the channel types using 4 folds. We save the training and testing indices as well as the epochs
    in order to be flexible for the later analyses.

    :param epochs:
    :param saving_directory:
    :return:
    """
    suf = ''
    # ----------- set the directories ----------
    saving_directory = op.join(config.SVM_path, subject)
    utils.create_folder(saving_directory)

    # ----------- load the epochs ---------------
    epochs = epoching_funcs.load_epochs_items(subject, cleaned=cleaned)
    epochs.pick_types(meg=True)

    # ----------- balance the position of the standard and the deviants -------
    # 'local' - Just make sure we have the same amount of standards and deviants for a given position. This may end up with
    #     1 standards/deviants for position 9 and 4 for the others.
    epochs_balanced = epoching_funcs.balance_epochs_violation_positions(epochs, balance_param="local")

    # ----------- do a sliding window to smooth the data if neeeded -------
    if sliding_window:
        epochs_balanced = epoching_funcs.sliding_window(epochs_balanced)
        suf += 'SW_'

    # =============================================================================================
    epochs_balanced_mag = epochs_balanced.copy().pick_types(meg='mag')
    epochs_balanced_grad = epochs_balanced.copy().pick_types(meg='grad')
    epochs_balanced_all_chans = epochs_balanced.copy().pick_types(meg=True)
    epochs_balanced_as_mag = epochs_balanced.copy().as_type('mag',mode = 'accurate')

    sensor_types = ['mag', 'grad', 'all_chans','as_mag']
    SVM_results = {'mag': [], 'grad': [], 'all_chans': [], 'as_mag': []}
    epochs_all = [epochs_balanced_mag, epochs_balanced_grad, epochs_balanced_all_chans,epochs_balanced_as_mag]
    # ==============================================================================================
    y_violornot = np.asarray(epochs_balanced.metadata['ViolationOrNot'].values)

    for l, senso in enumerate(sensor_types):
        epochs_senso = epochs_all[l]
        X_data = epochs_senso.get_data()
        All_SVM = []
        training_inds , testing_inds = train_test_different_blocks(epochs_senso,return_per_seq = False)
        for k in range(2):
            SVM_dec = SVM_decoder()
            SVM_dec.fit(X_data[training_inds[k],:,:], y_violornot[training_inds[k]])
            All_SVM.append(SVM_dec)
        SVM_results[senso] = {'SVM': All_SVM, 'train_ind': training_inds, 'test_ind': testing_inds,
                              'epochs': epochs_all[l]}
    np.save(op.join(saving_directory, suf + 'SVM_results.npy'), SVM_results)

# ______________________________________________________________________________________________________________________
def GAT_SVM_trained_all_sequences(subject,sliding_window=True,cleaned=True,metric = 'score'):
    """
    The SVM at a training times are tested at testing times. Allows to obtain something similar to the GAT from decoding.
    Dictionnary contains the GAT for each sequence separately. GAT_all contains the average over all the sequences
    :param SVM_results: output of generate_SVM_all_sequences
    :return: GAT averaged over the 4 classification folds
    """
    saving_directory = op.join(config.SVM_path, subject)
    # ----- build the right suffix to load the correct matrix -----
    suf = ''
    n_folds = 2
    if sliding_window:
        suf += 'SW_'

    # ---------- load the data ------------
    SVM_results = np.load(op.join(saving_directory, suf + 'SVM_results.npy'), allow_pickle=True).item()
    # ---------- load the habituation epochs -------
    epo_hab = epoching_funcs.load_epochs_items(subject, cleaned=cleaned)
    epo_hab = epo_hab["TrialNumber<11"]
    if sliding_window:
        epo_hab = epoching_funcs.sliding_window(epo_hab)

    # ----- initialize the results dictionnary ------
    GAT_sens_seq = {sens: [] for sens in config.ch_types}
    GAT_sens_seq_hab = {sens: [] for sens in config.ch_types}
    GAT_sens_seq_stand = {sens: [] for sens in config.ch_types}
    GAT_sens_seq_viol = {sens: [] for sens in config.ch_types}

    for sens in config.ch_types:
        GAT_all = []
        GAT_per_sens_and_seq = {'SeqID_%i' % i: [] for i in range(1, 8)}
        GAT_per_sens_and_seq_hab = {'SeqID_%i' % i: [] for i in range(1, 8)}
        GAT_per_sens_and_seq_stand = {'SeqID_%i' % i: [] for i in range(1, 8)}
        GAT_per_sens_and_seq_viol = {'SeqID_%i' % i: [] for i in range(1, 8)}

        epochs_sens = SVM_results[sens]['epochs']
        n_times = epochs_sens.get_data().shape[-1]
        SVM_sens = SVM_results[sens]['SVM']

        for sequence_number in range(1, 8):
            seqID = 'SeqID_%i' % sequence_number
            GAT_seq = np.zeros((n_folds, n_times, n_times))
            GAT_standard = np.zeros((n_folds, n_times, n_times))
            GAT_viol = np.zeros((n_folds, n_times, n_times))
            GAT_habituation = np.zeros((n_folds, n_times, n_times))

            for fold_number in range(n_folds):
                print("---- fold number %i -----"%fold_number)
                test_indices = SVM_results[sens]['test_ind'][fold_number]
                epochs_sens_test = epochs_sens[test_indices]
                epochs_sens_and_seq_test = epochs_sens_test["SequenceID == %i"%sequence_number]
                y_sens_and_seq_test = epochs_sens_and_seq_test.metadata["ViolationOrNot"].values
                data_seq_test = epochs_sens_and_seq_test.get_data()
                data_seq_hab = epo_hab["SequenceID == %i"%sequence_number].copy().pick_types(meg=sens).get_data()

                # Here split in standard and violations. Write another function that tests on habituations.
                inds_sens_and_seq_test_standard = np.where(epochs_sens_and_seq_test.metadata["ViolationOrNot"].values==0)[0]
                inds_sens_and_seq_test_violation = np.where(epochs_sens_and_seq_test.metadata["ViolationOrNot"].values==1)[0]

                if metric=='score':
                    GAT_seq[fold_number,:,:] = SVM_sens[fold_number].score(data_seq_test,y_sens_and_seq_test)
                    GAT_standard[fold_number,:,:] = SVM_sens[fold_number].score(data_seq_test[inds_sens_and_seq_test_standard],y_sens_and_seq_test[inds_sens_and_seq_test_standard])
                    GAT_viol[fold_number,:,:] = SVM_sens[fold_number].score(data_seq_test[inds_sens_and_seq_test_violation],y_sens_and_seq_test[inds_sens_and_seq_test_violation])
                    GAT_habituation[fold_number,:,:] = SVM_sens[fold_number].score(data_seq_hab,[0]*data_seq_hab.shape[0])
                else:
                    metric = 'projection_normal'
                    GAT_seq[fold_number,:,:] = np.mean(SVM_sens[fold_number].decision_function(data_seq_test[y_sens_and_seq_test==1]),axis=0) - np.mean(SVM_sens[fold_number].decision_function(data_seq_test[y_sens_and_seq_test!=1]),axis=0)
                    GAT_standard[fold_number,:,:] = np.mean(SVM_sens[fold_number].decision_function(data_seq_test[inds_sens_and_seq_test_standard]),axis=0)
                    GAT_viol[fold_number,:,:] = np.mean(SVM_sens[fold_number].decision_function(data_seq_test[inds_sens_and_seq_test_violation]),axis=0)
                    GAT_habituation[fold_number,:,:] = np.mean(SVM_sens[fold_number].decision_function(data_seq_hab),axis=0)
            #  --------------- now average across the folds ---------------
            GAT_seq_avg = np.mean(GAT_seq, axis=0)
            GAT_seq_habituation_avg = np.mean(GAT_habituation, axis=0)
            GAT_seq_standard_avg = np.mean(GAT_standard, axis=0)
            GAT_seq_viol_avg = np.mean(GAT_viol, axis=0)

            GAT_per_sens_and_seq[seqID] = GAT_seq_avg
            GAT_per_sens_and_seq_hab[seqID] = GAT_seq_habituation_avg
            GAT_per_sens_and_seq_stand[seqID] = GAT_seq_standard_avg
            GAT_per_sens_and_seq_viol[seqID] = GAT_seq_viol_avg

            GAT_all.append(GAT_seq_avg)

        GAT_sens_seq[sens] = GAT_per_sens_and_seq
        GAT_sens_seq_hab[sens] = GAT_per_sens_and_seq_hab
        GAT_sens_seq_stand[sens] = GAT_per_sens_and_seq_stand
        GAT_sens_seq_viol[sens] = GAT_per_sens_and_seq_viol

        GAT_sens_seq[sens]['average_all_sequences'] = np.mean(GAT_all, axis=0)
        times = epochs_sens_test.times

    if metric == 'score':
        GAT_results = {'GAT': GAT_sens_seq, 'times': times}
        np.save(op.join(saving_directory, suf + 'GAT_results.npy'), GAT_results)
        GAT_results_hab = {'GAT': GAT_sens_seq_hab, 'times': times}
        np.save(op.join(saving_directory, suf + 'GAT_results_hab.npy'), GAT_results_hab)
        GAT_results_stand = {'GAT': GAT_sens_seq_stand, 'times': times}
        np.save(op.join(saving_directory, suf + 'GAT_results_stand.npy'), GAT_results_stand)
        GAT_results_viol = {'GAT': GAT_sens_seq_viol, 'times': times}
        np.save(op.join(saving_directory, suf + 'GAT_results_viol.npy'), GAT_results_viol)
    else:
        GAT_results = {'projection_normal': GAT_sens_seq, 'times': times}
        np.save(op.join(saving_directory, suf + 'projection_normal_results.npy'), GAT_results)
        GAT_results_hab = {'projection_normal': GAT_sens_seq_hab, 'times': times}
        np.save(op.join(saving_directory, suf + 'projection_normal_results_hab.npy'), GAT_results_hab)
        GAT_results_stand = {'projection_normal': GAT_sens_seq_stand, 'times': times}
        np.save(op.join(saving_directory, suf + 'projection_normal_results_stand.npy'), GAT_results_stand)
        GAT_results_viol = {'projection_normal': GAT_sens_seq_viol, 'times': times}
        np.save(op.join(saving_directory, suf + 'projection_normal_results_viol.npy'), GAT_results_viol)

# ______________________________________________________________________________________________________________________
def unique_test_16(epochs_1st_sens, epochs_sens, test_indices):

    # epochs_sens = epoching_funcs.load_epochs_items(config.subjects_list[0])
    # test_indices = range(1,120)

    all_fields = []
    for m in test_indices:
        seqID_m = epochs_sens[m].metadata['SequenceID'].values[0]
        run_m = epochs_sens[m].metadata['RunNumber'].values[0]
        trial_number_m = epochs_sens[m].metadata['TrialNumber'].values[0]
        truple = (int(seqID_m), int(run_m), int(trial_number_m))
        all_fields.append(truple)
    unique_fields = list(set(all_fields))
    epochs_1st_item = []
    for seqID, run_number, trial_number in unique_fields:
        epochs_1st_item.append(epochs_1st_sens[
                                   'SequenceID == %i and RunNumber == %i and TrialNumber == %i ' % (
                                   seqID, run_number, trial_number)])

    return epochs_1st_item

# ______________________________________________________________________________________________________________________
def apply_SVM_filter_16_items_epochs(subject, times,sliding_window=True):
    """
    Function to apply the SVM filters built on all the sequences the 16 item sequences
    :param subject:
    :param times:the different times at which we want to apply the filter (if window is False). Otherwise (window = True),
    min(times) and max(times) define the time window on which we average the spatial filter.
    :param window: set to True if you want to average the spatial filter over a window.
    :return:

    """

    # ==== load the ems results ==============
    SVM_results_path = op.join(config.SVM_path, subject)
    suf = ''
    n_folds = 2
    if sliding_window:
        suf += 'SW_'
    SVM_results = np.load(op.join(SVM_results_path, suf + 'SVM_results.npy'), allow_pickle=True).item()
    # ==== define the paths ==============
    meg_subject_dir = op.join(config.meg_dir, subject)
    # ====== loading the 16 items sequences epoched on the first element ===================
    epochs_1st_element = epoching_funcs.load_epochs_full_sequence(subject, cleaned=False)
    if sliding_window:
        epochs_1st_element = epoching_funcs.sliding_window(epochs_1st_element)
    epochs_1st = epochs_1st_element["TrialNumber > 10"].as_type('mag',mode = 'accurate')
    sens = 'as_mag'
    # ====== compute the projections for each of the 3 types of sensors ===================
    print("---Now performing the analysis for sensor type %s -----"%sens)
    SVM_sens = SVM_results[sens]['SVM']
    epochs_sens = SVM_results[sens]['epochs']
    # = we initialize the metadata
    data_frame_meta = pd.DataFrame([])
    data_for_epoch_object = []
    # ===============================
    counter = 0
    for fold_number in range(n_folds):
        print('Fold ' + str(fold_number + 1) + ' on %i...' % n_folds)
        start = time.time()
        test_indices = SVM_results[sens]['test_ind'][fold_number]
        epochs_sens_test = epochs_sens[test_indices]
        points = epochs_sens_test.time_as_index(times)

        # ---- extract one unique 16 item sequence that may correspond to various test_indices ---
        epochs_1st_item_unique = unique_test_16(epochs_1st, epochs_sens, test_indices)
        for m, epochs_1st_sens_m in enumerate(epochs_1st_item_unique):
            data_1st_el_m = epochs_1st_sens_m.get_data()
            SVM_to_data = np.squeeze(SVM_sens[fold_number].decision_function(data_1st_el_m))
            epochs_1st_sens_m_filtered_data = np.mean(SVM_to_data[np.min(points):np.max(points), :], axis=0)
            data_for_epoch_object.append(np.squeeze(epochs_1st_sens_m_filtered_data))
            metadata_m = epochs_1st_sens_m.metadata
            metadata_m['SVM_filter_min_datapoint'] = np.min(points)
            metadata_m['SVM_filter_max_datapoint'] = np.max(points)
            metadata_m['SVM_filter_tmin_window'] = times[0]
            metadata_m['SVM_filter_tmax_window'] = times[-1]
            data_frame_meta = data_frame_meta.append(metadata_m)
            counter += 1
            print("Performing the projection for trial number %i from fold number %i \n"%(counter+1,fold_number+1))
        end = time.time()
        elapsed = end - start
        print('... lasted: ' + str(elapsed) + ' s')

        dat = np.expand_dims(np.asarray(data_for_epoch_object), axis=1)
        info = mne.create_info(['SVM'], epochs_1st.info['sfreq'])
        epochs_proj_sens = mne.EpochsArray(dat, info, tmin=-0.5)
        epochs_proj_sens.metadata = data_frame_meta
        suf += "_%i_%ims" % (int(np.min(times) * 1000), int(np.max(times) * 1000))
        epochs_proj_sens.save(meg_subject_dir + op.sep + sens + suf + '_SVM_on_16_items_test_window-epo.fif',
                              overwrite=True)

    return True

# ______________________________________________________________________________________________________________________
def apply_SVM_filter_16_items_epochs_habituation(subject, times=[x / 1000 for x in range(0, 750, 50)],
                                                  sliding_window=True):
    """
    Function to apply the SVM filters on the habituation trials. It is simpler than the previous function as we don't have to select the specific
    trials according to the folds.
    :param subject:
    :param times:
    :return:
    """

    # ==== load the ems results ==============
    SVM_results_path = op.join(config.SVM_path, subject)
    suf = ''
    n_folds = 2
    if sliding_window:
        suf += 'SW_'

    SVM_results = np.load(op.join(SVM_results_path, suf + 'SVM_results.npy'), allow_pickle=True).item()

    # ==== define the paths ==============
    meg_subject_dir = op.join(config.meg_dir, subject)
    epochs_1st_element = epoching_funcs.load_epochs_full_sequence(subject, cleaned=False)
    if sliding_window:
        epochs_1st_element = epoching_funcs.sliding_window(epochs_1st_element)
    # ====== loading the 16 items sequences epoched on the first element ===================

    epochs_1st = epochs_1st_element["TrialNumber < 11"].as_type('mag',mode = 'accurate')
    sens = 'as_mag'
    # ====== compute the projections for each of the 3 types of sensors ===================
    SVM_sens = SVM_results[sens]['SVM']
    points = SVM_results[sens]['epochs'][0].time_as_index(times)
    # = we initialize the metadata
    data_frame_meta = pd.DataFrame([])
    n_habituation = epochs_1st_element.get_data().shape[0]
    print('the number of habituation trials is %i'%n_habituation)
    # ========== les 4 filtres peuvent etre appliquees aux sequences d habituation sans souci, selection en fonction des indices ========
    data_1st_el_m = epochs_1st.get_data()
    epochs_1st_sens_filtered_data_folds = []
    for fold_number in range(n_folds):
        SVM_to_data = np.squeeze(SVM_sens[fold_number].decision_function(data_1st_el_m))
        print("The shape of SVM_to_data is ")
        print(SVM_to_data.shape)
        print(
            " === MAKE SURE THAT WHEN SELECTING SVM_to_data[point_of_interest,:] WE ARE INDEED CHOOSING THE TRAINING TIMES ===")
        epochs_1st_sens_filtered_data_folds.append(
            np.mean(SVM_to_data[:, np.min(points):np.max(points), :], axis=1))

    # ==== now that we projected the 4 filters, we can average over the 4 folds ================
    data_for_epoch_object = np.mean(epochs_1st_sens_filtered_data_folds, axis=0)

    metadata = epochs_1st.metadata
    print("==== the length of the epochs_1st_sens.metadata to append is %i ====" % len(metadata))
    metadata['SVM_filter_min_datapoint'] = np.min(points)
    metadata['SVM_filter_max_datapoint'] = np.max(points)
    metadata['SVM_filter_tmin_window'] = times[0]
    metadata['SVM_filter_tmax_window'] = times[-1]
    data_frame_meta = data_frame_meta.append(metadata)

    dat = np.expand_dims(data_for_epoch_object, axis=1)
    info = mne.create_info(['SVM'], epochs_1st.info['sfreq'])
    epochs_proj_sens = mne.EpochsArray(dat, info, tmin=-0.5)
    print("==== the total number of epochs is %i ====" % len(epochs_proj_sens))
    print("==== the total number of metadata fields is %i ====" % len(data_frame_meta))

    epochs_proj_sens.metadata = data_frame_meta

    suf += "_%i_%ims" % (int(np.min(times) * 1000), int(np.max(times) * 1000))

    epochs_proj_sens.save(meg_subject_dir + op.sep + sens + suf + '_SVM_on_16_items_habituation_window-epo.fif',
                          overwrite=True)

    return True

# =========================================================================================================
class SlidingWindow(TransformerMixin):
    """
    Aggregate time points in a "sliding window" manner

    Input: Anything x Anything x Time points
    Output - if averaging: Unchanged x Unchanged x Windows
    Output - if not averaging: Windows x Unchanged x Unchanged x Window size
                Note that in this case, the output may not be a real matrix in case the last sliding window is smaller than the others
    """

    # --------------------------------------------------
    def __init__(self, window_size, step, min_window_size=None, average=True, debug=False):
        """
        :param window_size: The no. of time points to average
        :param step: The no. of time points to slide the window to get the next result
        :param min_window_size: The minimal number of time points acceptable in the last step of the sliding window.
                                If None: min_window_size will be the same as window_size
        :param average: If True, just reduce the number of time points by averaging over each window
                        If False, each window is copied as-is to the output, without averaging
        """
        self._window_size = window_size
        self._step = step
        self._min_window_size = min_window_size
        self._average = average
        self._debug = debug

    # --------------------------------------------------
    # noinspection PyUnusedLocal
    def fit(self, x, y=None, *_):
        return self

    # --------------------------------------------------
    def transform(self, x):
        x = np.array(x)
        assert len(x.shape) == 3
        n1, n2, n_time_points = x.shape

        # -- Get the start-end indices of each window
        min_window_size = self._min_window_size or self._window_size
        window_start = np.array(range(0, n_time_points - min_window_size + 1, self._step))
        if len(window_start) == 0:
            # -- There are fewer than window_size time points
            raise Exception('There are only {:} time points, but at least {:} are required for the sliding window'.
                            format(n_time_points, self._min_window_size))
        window_end = window_start + self._window_size
        window_end[-1] = min(window_end[-1],
                             n_time_points)  # make sure that the last window doesn't exceed the input size

        if self._debug:
            win_info = [(s, e, e - s) for s, e in zip(window_start, window_end)]
            print('SlidingWindow transformer: the start,end,length of each sliding window: {:}'.
                  format(win_info))
            if len(win_info) > 1 and win_info[0][2] != win_info[-1][2] and not self._average:
                print(
                    'SlidingWindow transformer: note that the last sliding window is smaller than the previous ones, ' +
                    'so the result will be a list of 3-dimensional matrices, with the last list element having ' +
                    'a different dimension than the previous elements. ' +
                    'This format is acceptable by the RiemannDissimilarity transformer')

        if self._average:
            # -- Average the data in each sliding window
            result = np.zeros((n1, n2, len(window_start)))
            for i in range(len(window_start)):
                result[:, :, i] = np.mean(x[:, :, window_start[i]:window_end[i]], axis=2)

        else:
            # -- Don't average the data in each sliding window - just copy it
            result = []
            for i in range(len(window_start)):
                result.append(x[:, :, window_start[i]:window_end[i]])

        return result

