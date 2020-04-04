"""
This code conduct:
1. exclude a subset of data related to a certain illumination condition from an existing training set
2. keep a subset of data related to the same illumination condition from an existing test set

ATTENTION: this code require to run prepare_data_split.py first

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

import os
import os.path as ops


if __name__ == '__main__':
    batch_size = 8  # use to ignore the last for convenience

    # exclude subsets from train
    name_pattens_to_exclude = ['/00/', '/01/', '/06/', '/07/']
    output_folder = '../data_splits/illus_chg/'
    if not ops.exists(output_folder):
        os.makedirs(output_folder)

    lines_train = []
    json_file_path = "../data_splits/standard/train.json"
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

    lines_test = []
    json_file_path = "../data_splits/standard/test.json"
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