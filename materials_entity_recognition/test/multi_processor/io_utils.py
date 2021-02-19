import json
import os
import time
from pprint import pprint
import argparse
import sys

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif v.lower() in {'no', 'false', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2None(v):
    if v is None:
        return v
    if v.lower() in {'none', }:
        return None
    return v

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

def use_file_as_stdout(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    fw_out = open('{}.out'.format(file_path), 'w')
    sys.stdout = fw_out
    sys.stdout = Unbuffered(sys.stdout)
    fw_err = open('{}.err'.format(file_path), 'w')
    sys.stderr = fw_err
    sys.stderr = Unbuffered(sys.stderr)
    print('this is printed in the console')

def save_results(results, dir_path='../generated/results', prefix='results'):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_name = '{}_{}.json'.format(prefix, str(hash(str(results))))
    with open(os.path.join(dir_path, file_name), 'w') as fw:
        json.dump(results, fw, indent=2)
    return file_name

