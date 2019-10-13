

import sys
import os
import filecmp
import shutil


def compare_one_file_with_others():
    binary_directory_path = 'D:/iust_pdf_corpus/corpus_merged_coverage/coverage_temp/'
    coverage_sep_base_path = 'D:/iust_pdf_corpus/corpus_merged_coverage/coverage_sep/'

    binary_directory_path = 'D:/iust_pdf_corpus/corpus_merged_coverage/coverage_xml_temp/'
    coverage_sep_base_path = 'D:/iust_pdf_corpus/corpus_merged_coverage/coverage_xml_sep/'

    # file_name_1 = 'pdftc_010k_0002.coverage'
    # file_name_2 = 'pdftc_mozilla_0010.coverage'

    # x = filecmp.cmp(binary_directory_path+file_name_1, binary_directory_path+file_name_2, shallow=False)
    # print(x)

    # for file_name_1 in os.listdir(binary_directory_path):
    #     if filecmp.cmp(binary_directory_path+file_name_1, binary_directory_path+file_name_2, shallow=False):
    #         print(file_name_1)

    for file_name_1 in os.listdir(binary_directory_path):
        try:
            if not os.path.exists(coverage_sep_base_path + file_name_1):
                os.makedirs(coverage_sep_base_path + file_name_1)
            for file_name_2 in os.listdir(binary_directory_path):
                if file_name_1 != file_name_2:
                    if filecmp.cmp(binary_directory_path + file_name_1,
                                   binary_directory_path + file_name_2,
                                   shallow=False):
                        shutil.move(binary_directory_path + file_name_2,
                                    coverage_sep_base_path + file_name_1 + '/' + file_name_2)
                        print('Equal coverage found :)', file_name_2)
            shutil.move(binary_directory_path + file_name_1,
                        coverage_sep_base_path + file_name_1 + '/' + file_name_1)
        except:
            pass


def test():
    binary_directory_path = 'D:/iust_pdf_corpus/corpus_merged_coverage/coverage_temp/'
    file_name_1 = 'pdftc_010k_0001.coverage'
    file_name_2 = 'pdftc_mozilla_0010.coverage'
    # x = filecmp.cmp(binary_directory_path+file_name_1, binary_directory_path+file_name_2, shallow=False)
    # print(x)

    for file_name_1 in os.listdir(binary_directory_path):
        with open(binary_directory_path+file_name_1, 'rb') as f1:
            file1 = f1.read()
        for file_name_2 in os.listdir(binary_directory_path):
            with open(binary_directory_path+file_name_2, 'rb') as f2:
                file2 = f2.read()
            # print('file 1 len', len(file1))
            # print('file 2 len', len(file2))
            # print('-----------------------')
            if file_name_1 != file_name_2:
                if compare_2_file(file1, file2):
                    print('equal found!')


def compare_2_file(file1, file2):
    if len(file1) != len(file2):
        return False
    else:
        for i in range(len(file1)):
            if file1[i] != file2[i]:
                return False
    return True


def remove_empty_directories():
    base_path = 'D:/iust_pdf_corpus/corpus_merged_coverage/coverage_xml_sep/'
    all_directories = os.listdir(base_path)
    # print(all_directories)
    # print(os.listdir(base_path + all_directories[33]))
    for directory_name in all_directories:
        if os.listdir(base_path+directory_name)==[]:
            print('removing ', directory_name)
            os.rmdir(base_path+directory_name)
    print('end')


def main(argv):
    # compare_one_file_with_others()
    # test()
    remove_empty_directories()


if __name__ == "__main__":
    main(sys.argv)
