config = {
    'single_object_update': False,  # [False, True]
    'portion_of_rewrite_objects': 1/3.,  # [1/4., 1/3., 1/2.]
    'update_policy': 'random',  # ['random', 'top-down', 'bottom_up']
    'host1': './hosts/rawhost/host1.pdf',  # host1 path
    'host2': './hosts/rawhost/host2.pdf',  # host2 path
    'host3': './hosts/rawhost/host3.pdf',  # host3 path
    'raw_host_directory': './hosts/rawhost/',  # raw hosts path
    'new_host_directory': './iupdfs/',
    'number_of_host': 3,
    'iupdf_host1': './iupdfs/host1/',
    'iupdf_host2': './iupdfs/host2/',
    'iupdf_host3': './iupdfs/host3/',
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

# print(config['portion_of_rewrite_objects'])
# print(learning_config['dataset_size'][2])