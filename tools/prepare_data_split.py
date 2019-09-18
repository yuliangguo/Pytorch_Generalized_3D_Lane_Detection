#!/usr/bin/env python
import os
import os.path as ops
import sys
import random
import math
import glob

if __name__ == '__main__':
    dataset_dir = '/home/yuliangguo/Datasets/Apollo_Sim_3D_Lane_0917'
    json_file_list = glob.glob('{:s}/laneline*.json'.format(dataset_dir))
    batch_size = 8
    output_folder = '../data/sim3d_0917/'
    split_ratio = [0.8, 0.2]
    if not ops.exists(output_folder):
        os.makedirs(output_folder)

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
    # print('number of test sampels: ' + str(N - N1 - N2))

    # split int into train.txt val.txt test.txt,
    # output folder: baidu_all_data_runs/run_1, run_2, ...
    random.shuffle(lines)
    lines_train = lines[0:N1]
    with open(output_folder + '/train.json', 'w') as f:
        f.writelines("%s" % l for l in lines_train)
    f.close()

    lines_val = lines[N1:N1+N2]
    with open(output_folder + '/val.json', 'w') as f:
        f.writelines("%s" % l for l in lines_val)
    f.close()

    # lines_test = lines[N1+N2:]
    # with open(output_folder + '/test.json', 'w') as f:
    #     f.writelines("%s" % l for l in lines_test)
    # f.close()
