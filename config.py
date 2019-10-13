"""
ISUT-DeepFuzz global configuration file.
-- Before running IUST-DeepFuzz:
-- Set the settings such as dataset path and other configuration in this file.

"""

iu_config = {
    'single_object_update': False,  # [False, True]
    # The below option use only if 'single_object_update' set to False
    'portion_of_rewrite_objects': 1/5.,  # [1/4., 1/3., 1/2.] /**/ {host1_max: 1/5., host2_min: 1/3., host3_avg: 1/4.}
    #
    'update_policy': 'random',  # ['random', 'bottom_up', 'top-down'] /**/ {'bottom_up' for sou and 'random' for mou}
    'getting_object_policy': 'sequential',  # ['sequential', 'random']

    # Old pdf file path (same hosts)
    'number_of_hosts': 3,
    'raw_host_directory': 'incremental_update/hosts/rawhost_new/',  # raw hosts root directory path, update to new path 13970305
    'host1': 'incremental_update/hosts/rawhost_new/host1_max.pdf',  # host1 relative path
    'host2': 'incremental_update/hosts/rawhost_new/host2_min.pdf',  # host2 relative path
    'host3': 'incremental_update/hosts/rawhost_new/host3_avg.pdf',  # host3 relative path
    'host123': 'incremental_update/hosts/rawhost/host123.pdf',  # host123 full path

    # New generated/fuzzed objects path (by deep learning model)
    'baseline_object_path': 'incremental_update/hosts/baseline/baseline_obj_1193_from_testset_ii.txt',
    'new_objects_path': 'generated_results/pdfs/_newest_objects/',  # Not set yet
    'stream_directory_path': 'dataset/pdfs/small_size_dataset/binary_streams/',
    'stream_fuzzing_policy': 'basic_random',  # ['basic_random', 'other']

    # New pdf files by attaching above new pdf objects
    'new_pdfs_directory': 'incremental_update/new_pdfs/',   # new generated pdf file root directory
    'iupdf_host1': 'incremental_update/new_pdfs/host1/',
    'iupdf_host2': 'incremental_update/new_pdfs/host2/',
    'iupdf_host3': 'incremental_update/new_pdfs/host3/',

    # configuration setting to measure code coverage
    'sut_dir': 'D:/afl/mupdf/platform/win32/Release/',
    'sut_path': 'D:/afl/mupdf/platform/win32/Release/mutool.exe',
    'sut_arguments': ' clean -difa ',
    
    'visual_studio_developer_cmd_path':
        'C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\Team Tools\\Performance Tools\\',
}


learning_config = {
    'file_fromat': 'pdf',  # ['pdf', 'xml', 'html', 'png']
    'dataset_size': 'large',  # ['small', 'medium', 'large'] # switch to large dataset date: 1397-02-12
    'small_training_set_path': './dataset/pdfs/small_size_dataset/'
                               '06_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_'
                               'trainset_47711_shuffle_ii.txt',
    'small_validation_set_path': './dataset/pdfs/small_size_dataset/'
                                 '06_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_'
                                 'validationset_11927_shuffle_ii.txt',
    'small_testing_set_path': './dataset/pdfs/small_size_dataset/'
                                 '06_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_'
                                 'testset_35784_shuffle_ii.txt',

    'medium_training_set_path': './dataset/pdfs/medium_size_dataset/'
                               '07_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_'
                               'trainset_47711_shuffle_ii.txt',
    'medium_validation_set_path': './dataset/pdfs/medium_size_dataset/'
                                 '07_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_'
                                 'validationset_11927_shuffle_ii.txt',
    'medium_testing_set_path': './dataset/pdfs/medium_size_dataset/'
                                 '07_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_'
                                 'testset_35784_shuffle_ii.txt',

    'large_training_set_path': './dataset/pdfs/large_size_dataset/'
                                '05_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_'
                                'trainset_268371_shuffle_ii.txt',
    'large_validation_set_path': './dataset/pdfs/large_size_dataset/'
                                  '05_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_'
                                  'validationset_89457_shuffle_ii.txt',
    'large_testing_set_path': './dataset/pdfs/large_size_dataset/'
                               '05_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_'
                               'testset_119276_shuffle_ii.txt'
}


pdf_objects_config = {

}


pdf_corpus_config = {
    'corpus_root': 'D:/iust_pdf_corpus/corpus_garbage/',
    'pdf_dir1_path': 'D:/iust_pdf_corpus/corpus_garbage/all_00_10kb/',  # 'Set 1 of IUST corpus' ==> 0-10 kb
    'pdf_dir2_path': 'D:/iust_pdf_corpus/corpus_garbage/drive_deh_10_100kb/',  # 'Set 2 of IUST corpus' ==> 10-100 kb
    'pdf_dir3_path': 'D:/iust_pdf_corpus/corpus_garbage/drive_h_100_900kb/',  # 'Set 3 of IUST corpus' ==> 100-900 kb
    'pdf_dir4_path': 'D:/iust_pdf_corpus/corpus_garbage/mozilla/',  # 'Set 4 of IUST corpus' ==> mozilla corpus

    'corpus_merged': 'D:/iust_pdf_corpus/corpus_merged/',
    'corpus_merged_streams': 'D:/iust_pdf_corpus/corpus_merged_streams/'
}


# print(config['portion_of_rewrite_objects'])
# print(learning_config['dataset_size'][2])