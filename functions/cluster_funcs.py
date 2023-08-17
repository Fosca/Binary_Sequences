# This module contains all the functions that allow the computations to run on the cluster.
from __future__ import division
import sys
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_PUBLICATION/')
from functions import bids_funcs, preprocessing_funcs, utils, regression_funcs, SVM_funcs
import config

#_______________________________________________________________________________________________________________________
def create_qsub(function_name, folder_name, suffix_name, sublist_subjects=None, queue='Unicog_long'):
    import subprocess
    import os, glob

    # ==========================================================================================
    # ============= ============= create the jobs ============================= ============= ==
    # ==========================================================================================

    ########################################################################
    # List of parameters to be parallelized
    ListSubject = config.subjects_list
    if sublist_subjects is not None:
        ListSubject = sublist_subjects

    ########################################################################
    # Initialize job files and names

    List_python_files = []

    wkdir = config.cluster_path
    base_path = config.scripts_path
    initbody = 'import sys \n'
    initbody = initbody + "sys.path.append(" + "'" + base_path + "')\n"
    initbody = initbody + 'from functions import cluster_funcs\n'

    # Write actual job files
    python_file, Listfile, ListJobName = [], [], []

    for s, subject in enumerate(ListSubject):
        print(subject)

        additionnal_parameters = ''

        body = initbody + "cluster_funcs.%s('%s')" % (function_name, subject)

        jobname = suffix_name + '_' + subject

        ListJobName.append(jobname)

        # Write jobs in a dedicated folder
        path_jobs = wkdir + '/generated_jobs/' + folder_name + '/'
        utils.create_folder(path_jobs)
        name_file = path_jobs + jobname + '.py'
        Listfile.append(name_file)

        with open(name_file, 'w') as python_file:
            python_file.write(body)

    # ============== Loop over your jobs ===========

    jobs_path = config.cluster_path + "/generated_jobs/"
    results_path = config.cluster_path + "/results_qsub/"
    utils.create_folder(results_path + folder_name)
    list_scripts = sorted(glob.glob(jobs_path + folder_name + "/*.py"))

    # Loop over your jobs

    for i in list_scripts:
        # Customize your options here
        file_name = os.path.split(i)
        job_name = "%s" % file_name[1]

        walltime = "24:00:00"  # "24:00:00"
        memory = "10G"  # "24:00:00"
        if 'short' in queue:
            walltime = "2:00:00"  # "24:00:00"

        processors = "nodes=1:ppn=1"
        command = "python %s" % i
        standard_output = "/std_%s" % file_name[1]
        error_output = "/err_%s" % file_name[1]
        name_file = "/qsub_cmd_%s" % file_name[1]

        job_string = """#!/bin/bash
        #PBS -N %s
        #PBS -q %s 
        #PBS -l walltime=%s
        #PBS -l mem=%s
        #PBS -l %s
        #PBS -o %s
        #PBS -e %s 
        cd %s
        %s""" % (job_name, queue, walltime, memory, processors, results_path + folder_name + standard_output,
                 results_path + folder_name + error_output, results_path, command)

        # job_file = jobs_path + folder_name + '/' + name_file
        job_file = jobs_path + name_file
        fichier = open(job_file, "w")
        fichier.write(job_string)
        fichier.close()

        # Send job_string to qsub
        cmd = "qsub %s" % (job_file)
        subprocess.call(cmd, shell=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# DATA CONVERSION TO BIDS FORMAT
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# 0 ____________________________________________________________________________________________________
def convert_data_to_BIDS(subject):
    print("Converting to BIDS for subject %s "%subject)
    bids_funcs.convert_to_bids(subject)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# DATA PREPROCESSING
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1 ____________________________________________________________________________________________________
def filter_the_data(subject):
    preprocessing_funcs.run_filter(subject)
# 2 ____________________________________________________________________________________________________
def maxfilter_data(subject):
    preprocessing_funcs.run_maxwell_filter(subject)
# 3 ____________________________________________________________________________________________________
def ICA_data(subject):
    preprocessing_funcs.run_ica(subject)
# 4 ____________________________________________________________________________________________________
def indentify_components_ICA(subject):
    preprocessing_funcs.automatic_identification_of_components(subject)
# 5 ____________________________________________________________________________________________________
def apply_ICA(subject):
    preprocessing_funcs.apply_ica(subject)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# EPOCHING --- missing from this script as it should take the BIDS formatted data as input
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# REGRESSIONS FUNCTIONS (PREDICTORS : COMPLEXITY, SURPRISE ETC
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def linear_regressions(subject):
    print("--- APPENDING THE NEW PREDICTORS SURPRISE AND REPEAT/ALTERNATE INTO THE METADATA ---")
    regression_funcs.update_metadata_epochs_and_save_epochs(subject)

    config.noEEG = True
    filter_names = ['Hab','Stand','Viol']
    for filter_name in filter_names:
        print(" ----- ANALYSIS PRESENTED IN FIGURE 6B ------- ")
        regression_funcs.compute_regression(subject, ['Intercept','Complexity'], "", filter_name, remap_channels='grad_to_mag',apply_baseline=True,cleaned=True)

        print(" ----- ANALYSES PRESENTED IN FIGURE SX ------- ")
        regression_funcs.compute_regression(subject, ['Intercept', 'surprise_100', 'Surprisenp1', 'RepeatAlter', 'RepeatAlternp1'],
                                            "", filter_name, remap_channels='grad_to_mag',apply_baseline=True,cleaned=True)
        regression_funcs.compute_regression(subject, ['Complexity'], "/Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1/" + subject + "/residuals--remapped_gtmbaselined_clean-epo.fif",
                                            filter_name,cleaned=True)

        regression_funcs.compute_regression(subject, ['Intercept', 'Complexity', 'surprise_100', 'Surprisenp1', 'RepeatAlter', 'RepeatAlternp1'],
                                            "", filter_name, remap_channels='grad_to_mag', apply_baseline=True, cleaned=True)

# REGRESSIONS FUNCTIONS (PREDICTORS : COMPLEXITY, SURPRISE ETC
def linear_regressions(subject):
    print("--- APPENDING THE NEW PREDICTORS SURPRISE AND REPEAT/ALTERNATE and Complexity^2 INTO THE METADATA ---")
    regression_funcs.update_metadata_epochs_and_save_epochs(subject)

    config.noEEG = True
    filter_names = ['Hab','Stand','Viol']
    for filter_name in filter_names:
        print(" ----- ANALYSIS PRESENTED IN FIGURE 6B ------- ")
        regression_funcs.compute_regression(subject, ['Intercept','Complexity'], "", filter_name, remap_channels='grad_to_mag',apply_baseline=True,cleaned=True)

        print(" ----- ANALYSES PRESENTED IN FIGURE S6 ------- ")
        regression_funcs.compute_regression(subject, ['Intercept', 'surprise_100', 'Surprisenp1', 'RepeatAlter', 'RepeatAlternp1'],
                                            "", filter_name, remap_channels='grad_to_mag',apply_baseline=True,cleaned=True)

        regression_funcs.compute_regression(subject, ['Complexity'], "/Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1/" + subject + "/residuals--remapped_gtmbaselined_clean-epo.fif",
                                            filter_name,cleaned=True)



        regression_funcs.compute_regression(subject, ['Intercept', 'Complexity', 'surprise_100', 'Surprisenp1', 'RepeatAlter', 'RepeatAlternp1'],
                                            "", filter_name, remap_channels='grad_to_mag', apply_baseline=True, cleaned=True)



def linear_regressions_Review(subject):
    print("--- APPENDING THE NEW PREDICTORS SURPRISE AND REPEAT/ALTERNATE and Complexity^2 INTO THE METADATA ---")
    regression_funcs.update_metadata_epochs_and_save_epochs(subject)

    config.noEEG = True

    filter_names = ['Stand','Viol']
    for filter_name in filter_names:

        # complexity and complexity squared are very correlated. A way to avoid this issue is to perform hierarchical regressions, linearly regressing complexity squared on the residuals
        regression_funcs.compute_regression(subject, ['Intercept', 'Complexity','surprise_100', 'Surprisenp1', 'RepeatAlter', 'RepeatAlternp1'],
                                            "", filter_name, remap_channels='grad_to_mag', apply_baseline=True, cleaned=True)

        regression_funcs.compute_regression(subject, ['Complexity_squared'], "/Intercept_Complexity_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1/" + subject + "/residuals--remapped_gtmbaselined_clean-epo.fif",
                                            filter_name,cleaned=True)
        # print(" ----- ANALYSIS PRESENTED IN FIGURE 6B ------- ")
        # regression_funcs.compute_regression(subject, ['Intercept','Complexity'], "", filter_name, remap_channels='grad_to_mag',apply_baseline=True,cleaned=True)
        #
        # print(" ----- ANALYSES PRESENTED IN FIGURE SX ------- ")
        # regression_funcs.compute_regression(subject, ['Intercept', 'surprise_100', 'Surprisenp1', 'RepeatAlter', 'RepeatAlternp1'],
        #                                     "", filter_name, remap_channels='grad_to_mag',apply_baseline=True,cleaned=True)
        #
        # regression_funcs.compute_regression(subject, ['Complexity'], "/Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1/" + subject + "/residuals--remapped_gtmbaselined_clean-epo.fif",
        #                                     filter_name,cleaned=True)
        #
        # regression_funcs.compute_regression(subject, ['Intercept', 'Complexity', 'surprise_100', 'Surprisenp1', 'RepeatAlter', 'RepeatAlternp1'],
        #                                     "", filter_name, remap_channels='grad_to_mag', apply_baseline=True, cleaned=True)

        # regression_funcs.compute_regression(subject, ['Intercept', 'Complexity','Complexity_squared','surprise_100', 'Surprisenp1', 'RepeatAlter', 'RepeatAlternp1'],
        #                                     "", filter_name, remap_channels='grad_to_mag', apply_baseline=True, cleaned=True)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# DECODING - TRAINING THE DECODERS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def train_SVM_decoder_all_sequences(subject):
    SVM_funcs.generate_SVM_all_sequences(subject)

def test_SVM_decoder_on_each_sequence(subject):
    SVM_funcs.GAT_SVM_trained_all_sequences(subject,metric='projection_normal')
    SVM_funcs.GAT_SVM_trained_all_sequences(subject,metric='score')

def apply_decoder_on_full_sequences(subject):
    SVM_funcs.apply_SVM_filter_16_items_epochs(subject, times=[0.131, 0.210], sliding_window=True, cleaned=True)
    SVM_funcs.apply_SVM_filter_16_items_epochs_habituation(subject, times=[0.131, 0.210], sliding_window=True, cleaned=True)
