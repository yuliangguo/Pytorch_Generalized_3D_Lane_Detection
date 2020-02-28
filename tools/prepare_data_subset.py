#!/usr/bin/env python
import os
import os.path as ops
import sys
import random
import math
import glob

if __name__ == '__main__':

    # exclude subsets from train
    # val_name_pattens = ['/18/', '/19/', '/20/', '/21/', '/22/', '/23/']
    # name_pattens_to_exclude = ['/02/', '/03/', '/08/', '/09/']
    name_pattens_to_exclude = ['/00/', '/01/', '/06/', '/07/']
    batch_size = 8
    output_folder = '../data/sim3d_0924_exclude_daytime/'
    if not ops.exists(output_folder):
        os.makedirs(output_folder)

    lines_train = []
    json_file_path = "../data/sim3d_0924_random_split/train.json"
    assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)
    with open(json_file_path) as f:
        lines_i = f.readlines()
    f.close()

    for line in lines_i:
        to_discard = False
        for name_patten in name_pattens_to_exclude:
            if name_patten in line:
                to_discard = True
                break

        if not to_discard:
            lines_train.append(line)

    lines_train = lines_train[:len(lines_train)//batch_size*batch_size]

    with open(output_folder + '/train.json', 'w') as f:
        f.writelines("%s" % l for l in lines_train)
    f.close()

    #########################################################################################
    # include subsets in test
    name_pattens_to_include = ['/00/', '/01/', '/06/', '/07/']
    batch_size = 8
    output_folder = '../data/sim3d_0924_exclude_daytime/'
    if not ops.exists(output_folder):
        os.makedirs(output_folder)

    lines_test = []
    json_file_path = "../data/sim3d_0924_random_split/val.json"
    assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)
    with open(json_file_path) as f:
        lines_i = f.readlines()
    f.close()

    for line in lines_i:
        to_discard = False
        for name_patten in name_pattens_to_include:
            if name_patten in line:
                lines_test.append(line)

    lines_test = lines_test[:len(lines_test) // batch_size * batch_size]

    with open(output_folder + '/test.json', 'w') as f:
        f.writelines("%s" % l for l in lines_test)
    f.close()