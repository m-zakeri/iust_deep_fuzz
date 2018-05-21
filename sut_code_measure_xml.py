"""
Script to calculate the percentage of basic block code coverage
for each PDF file in IUST_PDF_CORPUS.
Then sort PDF files by the amount of their code coverage
and return top 'n' most coverage file as python dictionary.

"""
import sys
import os
import csv
import operator
from xml.dom.minidom import parse
import xml.dom.minidom
import xml.etree.ElementTree as et


def calculate_covered_block_percent():
    xmls_directory_path = 'D:/afl/mupdf/platform/win32/Release/all_00_10kb_coverage_xml/'
    # Path full merged
    xmls_directory_path = 'D:/iust_pdf_corpus/corpus_merged_coverage/coverage_xml/'
    file_name_coverage_dic = {}
    for file_name in os.listdir(xmls_directory_path):
        tree = et.parse(xmls_directory_path + file_name)
        root = tree.getroot()
        blocks_covered = int(root[0][6].text)
        blocks_not_covered = int(root[0][7].text)
        blocks_total = blocks_covered + blocks_not_covered
        block_coverage_percent = (blocks_covered * 100) / blocks_total

        print('blocks_covered=', blocks_covered)
        print('blocks_not_covered=', blocks_not_covered)
        print('blocks_total=', blocks_total)
        print('block_coverage_percent', block_coverage_percent)

        file_name_coverage_dic.update({file_name: block_coverage_percent})

    print('Dictionary created. Sorting dictionary...')
    file_name_coverage_dic_sorted = sorted(file_name_coverage_dic.items(), key=operator.itemgetter(1))
    file_name_coverage_dic_sorted = dict(file_name_coverage_dic_sorted)

    print('Saving to csv file...')
    with open('pdf_file_name__block_coverage_full_6160_file.csv', 'w', newline='') as f:
        fieldnames = ['pdf_file_name', 'block_coverage']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        data = [dict(zip(fieldnames, [k, v])) for k, v in file_name_coverage_dic_sorted.items()]
        writer.writerows(data)

    print('work finished!')


def get_statistical_info_of_coverd_blocks():
    """Read a CSV file using csv.DictReader"""
    csv_path = 'seed/pdf_file_name__block_coverage_full_6160_file.csv'
    coverage_percent_list = []
    print('Reading csv file and retrieve coverage percents ...')
    sum = 0
    index = 0
    with open(csv_path, 'r') as f:
        # reader = csv.reader(f)
        reader = csv.DictReader(f, delimiter=',')
        for line in reader:
            # print(line["object_id"]),
            # print(line["object_len"])
            coverage_percent_list.append(float(line['block_coverage']))
            sum += float(line['block_coverage'])
            index += 1
    print('sum=', sum)
    print('index=', index)
    print('average=', sum/index)
    """
    sum = 62310.35054070596
    index = 6160
    average = 10.1153166462185
    """


def main(argv):
    # calculate_covered_block_percent()
    get_statistical_info_of_coverd_blocks()


if __name__ == "__main__":
    main(sys.argv)

