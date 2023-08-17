import initialization_paths
from functions import cluster_funcs

print('--- --- --- send on the cluster --- --- --- ')

#cluster_funcs.create_qsub('train_SVM_decoder_all_sequences', 'create_decoder', 'create_decoder', queue='Nspin_run32')
#cluster_funcs.create_qsub('test_SVM_decoder_on_each_sequence', 'test_SVM_single_item', 'test_SVM_single_item', queue='Nspin_run32')
#cluster_funcs.create_qsub('apply_decoder_on_full_sequences', 'test_SVM_16', 'test_SVM_16', queue='Nspin_run32')
cluster_funcs.create_qsub('linear_regressions_Review', 'R2_lin_reg', 'R2_lin_reg', queue='Nspin_run32')

#cluster_funcs.create_qsub('convert_data_to_BIDS', 'BIDS', 'BIDS', queue='Nspin_long')

print('--- --- --- finished sending on the cluster --- --- --- ')

