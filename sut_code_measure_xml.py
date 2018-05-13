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
    with open('pdf_file_name__block_coverage.csv', 'w', newline='') as f:
        fieldnames = ['pdf_file_name', 'block_coverage']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        data = [dict(zip(fieldnames, [k, v])) for k, v in file_name_coverage_dic_sorted.items()]
        writer.writerows(data)

    print('work finished!')


def main(argv):
    calculate_covered_block_percent()


if __name__ == "__main__":
    main(sys.argv)

