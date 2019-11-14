#!/usr/bin/env python
import os
import os.path as ops
import sys
import random
import math
import glob

if __name__ == '__main__':

    # val_name_pattens = ['/18/', '/19/', '/20/', '/21/', '/22/', '/23/']
    val_name_pattens = ['/02/', '/08/']
    batch_size = 8
    dataset_dir = '/home/yuliangguo/Datasets/Apollo_Sim_3D_Lane_0924'
    json_file_list = glob.glob('{:s}/laneline*.json'.format(dataset_dir))
    output_folder = '../data/sim3d_0924/'
    if not ops.exists(output_folder):
        os.makedirs(output_folder)

    lines_train = []
    lines_val = []
    for json_file_path in json_file_list:
        assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)
        with open(json_file_path) as f:
            lines_i = f.readlines()
        f.close()

        for line in lines_i:
            is_val = False
            for name_patten in val_name_pattens:
                if name_patten in line:
                    is_val = True
                    break

            if is_val:
                lines_val.append(line)
            else:
                lines_train.append(line)

    lines_train = lines_train[:len(lines_train)//batch_size*batch_size]
    lines_val = lines_val[:len(lines_val)//batch_size*batch_size]

    with open(output_folder + '/train.json', 'w') as f:
        f.writelines("%s" % l for l in lines_train)
    f.close()

    with open(output_folder + '/val.json', 'w') as f:
        f.writelines("%s" % l for l in lines_val)
    f.close()

