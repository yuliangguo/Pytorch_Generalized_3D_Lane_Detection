"""
1. A standard five-fold split of training and testing data sets.
2. A subset of rarely observed examples are extracted as another test set
ATTENTION: if you rerun this, a new random split will replace the old one. This will cause discrepancy between
           the new test ground-truth and previous prediction on the test data

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

import os
import os.path as ops
import random
import math
import glob
import shutil

if __name__ == '__main__':
    dataset_dir = '~/Datasets/Apollo_Sim_3D_Lane_Release'
    json_file_list = glob.glob('{:s}/laneline*.json'.format(dataset_dir))
    batch_size = 8
    output_folder = '../data_splits/standard/'
    output_folder_subset = '../data_splits/rare_subset/'
    split_ratio = [0.8, 0.2]
    if not ops.exists(output_folder):
        os.makedirs(output_folder)
    if not ops.exists(output_folder_subset):
        os.makedirs(output_folder_subset)

    lines = []
    for json_file_path in json_file_list:
        assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)
        with open(json_file_path) as f:
            lines_i = f.readlines()
        f.close()
        lines = lines + lines_i

    N = len(lines)
    print('total number of sampels: ' + str(N))
    N1 = int(math.floor(N * split_ratio[0]//batch_size*batch_size))
    print('number of train sampels: ' + str(N1))
    N2 = int(math.floor(N * split_ratio[1]//batch_size*batch_size))
    print('number of val sampels: ' + str(N2))

    # split int into train.txt val.txt test.txt,
    random.shuffle(lines)
    lines_train = lines[0:N1]
    with open(output_folder + '/train.json', 'w') as f:
        f.writelines("%s" % l for l in lines_train)
    f.close()

    lines_val = lines[N1:N1+N2]
    with open(output_folder + '/val.json', 'w') as f:
        f.writelines("%s" % l for l in lines_val)
    f.close()

    """
        extract a subset of hard samples
    """
    with open(output_folder + '/test.json') as f:
        lines_val = f.readlines()
    f.close()

    with open(output_folder_subset + '/test.json', 'w') as f:
        f.writelines("%s" % l for l in lines_val if ('/06/' in l or '/07/' in l or '/08/' in l or '/09/' in l or '/10/' in l or '/11/' in l))
    f.close()

    # copy the same training set
    shutil.copyfile(output_folder + '/train.json', output_folder_subset + '/train.json')

