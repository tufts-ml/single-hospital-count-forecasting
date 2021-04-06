'''
arg_types.py
------------
Checks that filenames specified by command-line arguments have proper suffixes.
'''

import argparse

def csv_file(text):
        if not text.endswith('.csv'):
            raise argparse.ArgumentTypeError('Not a valid csv file name.')
        return text

def json_file(text):
    if not text.endswith('.json'):
        raise argparse.ArgumentTypeError('Not a valid json file name.')
    return text

def pdf_file(text):
    if not text.endswith('.pdf'):
        raise argparse.ArgumentTypeError('Not a valid pdf file name.')
    return text