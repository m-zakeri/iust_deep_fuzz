iu_config = {
    'single_object_update': False,  # [False, True]
    'portion_of_rewrite_objects': 1/3.,  # [1/4., 1/3., 1/2.]
    'update_policy': 'random',  # ['random', 'top-down', 'bottom_up']

    # Old pdf file path (same hosts)
    'number_of_hosts': 3,
    'raw_host_directory': './hosts/rawhost/',  # raw hosts root directory path
    'host1': './hosts/rawhost/host1.pdf',  # host1 full path
    'host2': './hosts/rawhost/host2.pdf',  # host2 full path
    'host3': './hosts/rawhost/host3.pdf',  # host3 full path
    'host123': './hosts/rawhost/host123.pdf',  # host123 full path

    # New generated/fuzzed objects path (by deep learning model)
    'new_objects_path': 'not set yet',
    'stream_directory_path': './dataset/pdfs/small_size_dataset/binary_streams/',

    # New pdf files by attaching above new pdf objects
    'new_pdfs_directory': './new_pdfs/',   # new generated pdf file root directory
    'iupdf_host1': './new_pdfs/host1/',
    'iupdf_host2': './new_pdfs/host2/',
    'iupdf_host3': './new_pdfs/host3/',

    # configuration setting to measure code coverage
    'sut_dir': 'D:/afl/mupdf/platform/win32/Release/',
    'sut_path': 'D:/afl/mupdf/platform/win32/Release/mutool.exe',
    'sut_arguments': ' clean -difa ',
    
    'visual_studio_developer_cmd_path':
        'C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\Team Tools\\Performance Tools\\',
}


learning_config = {
    'file_fromat': 'pdf',  # ['pdf', 'xml', 'html', 'png']
    'dataset_size': 'small',  # ['small', 'medium', 'large']
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
                               'testset_119276_shuffle_ii.txt',

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