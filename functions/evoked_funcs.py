import config
import os.path as op
import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import sem



def load_evoked(subject='all', filter_name='', filter_not=None, root_path=None, cleaned=True, evoked_resid=False):
    """
    Cette fonction charge tous les evoques ayant un nom qui commence par filter_name et n'ayant pas filter_not dans leur nom.
    Elle cree un dictionnaire ayant pour champs les differentes conditions
    :param subject: 'all' si on veut charger les evoques de tous les participants de config.subject_list sinon mettre le nom du participant d'interet
    :param filter_name: element du nom de fichier à inclure
    :param filter_not: element du nom de fichier à exclure
    :param root_path: dossier source (défaut data/MEG/sujet)
    :param cleaned: chercher dans le dossier evoked_cleaned (evoked à partir d'epochs après autocorrect)
    :param evoked_resid: chercher dans le dossier evoked_resid (evoked à partir d'epochs après regression surprise) !! surpasse argument "cleaned" !!
    :return: 
    """

    import glob
    evoked_dict = {}
    if subject == 'all':
        for subj in config.subjects_list:

            if config.noEEG:
                sub_path = op.join(config.meg_dir, subj, 'noEEG')
            else:
                sub_path = op.join(config.meg_dir, subj)

            if cleaned:
                path_evo = op.join(sub_path, 'evoked_cleaned')
            else:
                path_evo = op.join(sub_path, 'evoked')
            if evoked_resid:
                path_evo = op.join(sub_path, 'evoked_resid')
            if root_path is not None:
                path_evo = op.join(root_path, subj)
            evoked_names = sorted(glob.glob(path_evo + op.sep + filter_name + '*.fif'))
            file_names = []
            full_names = []
            for names in evoked_names:
                path, file = op.split(names)
                if filter_name in names:
                    if filter_not is not None:
                        if filter_not not in names:
                            file_names.append(file)
                            full_names.append(names)
                    else:
                        file_names.append(file)
                        full_names.append(names)

            print(path_evo)
            print(file_names)
            for k in range(len(file_names)):
                if file_names[k][:-7] in evoked_dict.keys():
                    evoked_dict[file_names[k][:-7]].append(mne.read_evokeds(full_names[k]))
                else:
                    evoked_dict[file_names[k][:-7]] = [mne.read_evokeds(full_names[k])]
    else:
        if config.noEEG:
            sub_path = op.join(config.meg_dir, subject, 'noEEG')
        else:
            sub_path = op.join(config.meg_dir, subject)

        if cleaned:
            path_evo = op.join(sub_path, 'evoked_cleaned')
        else:
            path_evo = op.join(sub_path, 'evoked')
        if evoked_resid:
            path_evo = op.join(sub_path, 'evoked_resid')
        if root_path is not None:
            path_evo = op.join(root_path, subject)
        evoked_names = glob.glob(path_evo + op.sep + filter_name + '*.fif')
        file_names = []
        full_names = []
        for names in evoked_names:
            path, file = op.split(names)
            if filter_name in names:
                if filter_not is not None:
                    if filter_not not in names:
                        file_names.append(file)
                        full_names.append(names)
                else:
                    file_names.append(file)
                    full_names.append(names)

        evoked_dict = {file_names[k][:-7]: mne.read_evokeds(full_names[k]) for k in range(len(file_names))}

    return evoked_dict, path_evo


def load_regression_evoked(subject='all', path='', subpath='',filter =''):
    """
    Load evoked for several subjects when paths are in the format "path + subject + subpath"
    /!\ all files in the folder are loaded (alphabetical order (?))
    """

    import glob
    evoked_dict = {}
    if subject == 'all':
        for subj in config.subjects_list:
            subject_path = op.join(path, subj, subpath)
            evoked_names = sorted(glob.glob(subject_path + op.sep + '*'+filter+'*.fif'))
            # evoked_dict = {'evo'+str(k): mne.read_evokeds(evoked_names[k]) for k in range(len(evoked_names))}

            file_names = []
            full_names = []
            for names in evoked_names:
                tmppath, file = op.split(names)
                file_names.append(file)
                full_names.append(names)
            print(file_names)
            for k in range(len(file_names)):
                if file_names[k][:-8] in evoked_dict.keys():
                    evoked_dict[file_names[k][:-8]].append(mne.read_evokeds(full_names[k]))
                else:
                    evoked_dict[file_names[k][:-8]] = [mne.read_evokeds(full_names[k])]

    return evoked_dict


def plot_evoked_with_sem_1cond(data, cond, ch_type, ch_inds, color=None, filter=True, axis=None):
    times = data[0][0].times * 1000

    group_data_seq = []
    for nn in range(len(data)):
        sub_data = data[nn][0].copy()
        if ch_type == 'eeg':
            sub_data = np.array(sub_data.pick_types(meg=False, eeg=True)._data)
        elif ch_type == 'mag':
            sub_data = np.array(sub_data.pick_types(meg='mag', eeg=False)._data)
        elif ch_type == 'grad':
            sub_data = np.array(sub_data.pick_types(meg='grad', eeg=False)._data)
        if np.size(ch_inds) > 1:
            group_data_seq.append(sub_data[ch_inds].mean(axis=0))
        else:
            group_data_seq.append(sub_data[ch_inds])

    mean = np.mean(group_data_seq, axis=0)
    ub = mean + sem(group_data_seq, axis=0)
    lb = mean - sem(group_data_seq, axis=0)

    if filter == True:
        mean = savgol_filter(mean, 9, 3)
        ub = savgol_filter(ub, 9, 3)
        lb = savgol_filter(lb, 9, 3)

    if axis is not None:
        axis.fill_between(times, ub, lb, color=color, alpha=.2)
        axis.plot(times, mean, color=color, linewidth=1.5, label=cond)
    else:
        plt.fill_between(times, ub, lb, color=color, alpha=.2)
        plt.plot(times, mean, color=color, linewidth=1.5, label=cond)
