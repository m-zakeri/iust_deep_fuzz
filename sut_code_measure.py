"""
Script to measure:
- code coverage
- pass rate
for different part of mupdf library include mutool and mupdf viewer
Note: code_coverage() function departed and not work.
For measure code coverage I write a windows batch file that is works nice.
"""

__version__ = '0.1'
__author__ = 'Morteza'

import sys
import subprocess
import os

from config import iu_config
from config import pdf_corpus_config


def code_coverage():
    sys.path.append(iu_config['visual_studio_developer_cmd_path'])
    print('path set.')
    print(sys.path, end='\n')

    cmd = 'start VSPerfMon /coverage /output:mytestrun_v4.coverage'
    subprocess.Popen(cmd, shell=True)
    print('cmd1 finished.')

    cmd = iu_config['sut_path'] + iu_config['sut_arguments'] + iu_config['sut_dir'] + '_pdfs/2.pdf'
    subprocess.run(cmd, shell=True)
    print('cmd2 finished.')

    cmd = 'VSPerfCmd /shutdown'
    subprocess.Popen(cmd, shell=True)
    print('cmd3 finished.')


def pass_rate_by_check_return_code():
    passed_pdfs = 0
    total_pdfs = 0
    for filename in os.listdir(pdf_corpus_config['corpus_merged']):
        cmd = iu_config['sut_path'] + iu_config['sut_arguments'] + pdf_corpus_config['corpus_merged'] + filename
        # cmd = 'D:/afl/mupdf/platform/win32/Release/mutool.exe clean -difa D:/iust_pdf_corpus/corpus_merged/pdftc_mozilla_0333.pdf'
        print(cmd)
        x = subprocess.run(cmd, shell=True)

        # print(x.stderr)
        # print(20*'---')
        # print(x.returncode)

        total_pdfs += 1
        if x.returncode == 0:
            passed_pdfs += 1
        # input()

    print('passed_pdfs: %s out of %s' % (passed_pdfs, total_pdfs))
    print('pass_rate percentage:', (100 * passed_pdfs)/total_pdfs)


def pass_rate_by_check_output_size():
    passed_pdfs = 0
    total_pdfs = 0
    for filename in os.listdir(pdf_corpus_config['corpus_merged']):
        cmd = iu_config['sut_path'] + iu_config['sut_arguments'] + pdf_corpus_config['corpus_merged'] + filename
        # cmd = 'D:/afl/mupdf/platform/win32/Release/mutool.exe clean -difa D:/iust_pdf_corpus/corpus_merged/pdftc_mozilla_0333.pdf'
        print(cmd)
        x = subprocess.run(cmd,check=False, shell=True)
        total_pdfs += 1
        print(os.path.getsize('out.pdf'))
        if os.path.getsize('out.pdf') > 0:
            passed_pdfs += 1
        # if x.stderr is None:
        #     passed_pdfs += 1
        print(x.stderr)
        input()

    print('passed_pdfs: %s out of %s' % (passed_pdfs, total_pdfs))
    print('pass_rate percentage:', (100 * passed_pdfs) / total_pdfs)


def set_cwd(path):
    """ set the current python working directory to costume path"""
    os.chdir(path)


def main(argv):
    pass_rate_by_check_output_size()


if __name__ == "__main__":
    main(sys.argv)

