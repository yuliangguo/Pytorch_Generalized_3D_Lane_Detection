#!/usr/bin/env python
"""
Process apollo sim laneline raw labels, and save in .json file
"""
import argparse
import os
import os.path as ops
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# color coding of different lane labels
colors = np.array([[0, 0, 0], [60, 10, 0], [120, 20, 0], [180, 30, 0], [240, 40, 0],
                   [0, 250, 0], [0, 40, 190], [0, 30, 130], [0, 20, 70], [0, 10, 10],
                   [0, 0, 0], [125, 0, 125], [125, 250, 125]], dtype=np.uint16)
colors = np.reshape(colors, [-1, 3])

img_height = 1080
img_width = 1920
K = np.array([[2015.0,      0, 960.0],
             [      0, 2015.0, 540.0],
             [      0,      0,     1]])
vis = False


def get_lists(test_file):
    with open(test_file) as f:
        test_list = f.readlines()

    image_list = []
    label_list = []
    name_list = []
    for line in test_list:
        line = line.replace("\n", "")
        line = line.replace("./", "")
        image_list.append("images/" + line)
        label_list.append("labels/" + line[:-4] + ".txt")
        name_list.append(line[:-4].replace("/", "_"))
    return image_list, label_list, name_list


def laneline_label_generator(base_folder, image_name, label_name, output_gt_file):
    img = cv2.imread(base_folder + image_name)
    # label_img = np.zeros((img_height, img_width), dtype=np.int8)

    # extract lanes and types from ground-truth label file
    centerlines, lanelines, cam_height, cam_pitch = process_lane_label_apollo_sim_3D(base_folder + label_name)

    # compute projection matrix
    proj_g2c = np.array([[1,                             0,                              0,          0],
                         [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                         [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0],
                         [0,                             0,                              0,          1]])
    proj_c2g = np.linalg.inv(proj_g2c)
    centerlines_g = []
    lanelines_g = []

    # visualize laneline by projecting to image
    for i, lane3D in enumerate(centerlines):
        lane3D = np.array(lane3D)
        # project laneline 3D in camera coordinates to ground coordinates
        centerline = np.concatenate([lane3D, np.ones([lane3D.shape[0], 1])], axis=1)
        centerline_g = np.matmul(centerline, proj_c2g.T)
        centerlines_g.append(centerline_g[:, :3].tolist())

        # project to image (camera coordinates to image coordinates)
        lane2D = np.matmul(lane3D, K.T)
        lane2D = np.divide(lane2D, np.expand_dims(lane2D[:, 2], -1))
        # draw on image
        for j in range(1, lane2D.shape[0]):
            img = cv2.line(img,
                           (int(lane2D[j-1, 0]), int(lane2D[j-1, 1])),
                           (int(lane2D[j, 0]), int(lane2D[j, 1])),
                           color=[255, 0, 0])

    for i, lane3D in enumerate(lanelines):
        lane3D = np.array(lane3D)
        # project laneline 3D in camera coordinates to ground coordinates
        laneline = np.concatenate([lane3D, np.ones([lane3D.shape[0], 1])], axis=1)
        laneline_g = np.matmul(laneline, proj_c2g.T)
        lanelines_g.append(laneline_g[:, :3].tolist())
        # project to image (camera coordinates to image coordinates)
        lane2D = np.matmul(np.array(lane3D), K.T)
        lane2D = np.divide(lane2D, np.expand_dims(lane2D[:, 2], -1))
        # draw on image
        for j in range(1, lane2D.shape[0]):
            img = cv2.line(img,
                           (int(lane2D[j-1, 0]), int(lane2D[j-1, 1])),
                           (int(lane2D[j, 0]), int(lane2D[j, 1])),
                           color=[0, 255, 0])

    # generate json
    result = {}
    valid_img = True
    result["raw_file"] = image_name
    result['cam_height'] = cam_height
    result['cam_pitch'] = cam_pitch
    result["centerLines"] = np.asarray(centerlines_g).tolist()
    result["laneLines"] = np.asarray(lanelines_g).tolist()

    with open(output_gt_file, "a") as outfile:
        outfile.write(json.dumps(result))
        outfile.write("\n")
        outfile.close()

    return img, valid_img


def process_lane_label_apollo_sim_3D(label_file):
    """
    Process lane ground-truth file to output a list of lanes and a list of lane types
    :param label_file: the .lane file path saving ground-truth lanes
    :param w_org: width of original image
    :param h_org: height of original image
    :return: a list of lanes, a list of lane types
    """

    with open(label_file, 'r') as jf:
        lane_data = json.load(jf)

    centerlanes_in = lane_data['laneList']
    lanelines_in = lane_data['laneBoundaryList']

    centerlanes_out = []
    for centerlane_in in centerlanes_in:
        centerlane_out = []
        for pt_3d in centerlane_in['pos3DInCameraList']:
            centerlane_out.append([pt_3d['x'], pt_3d['y'], pt_3d['z']])
        centerlanes_out.append(centerlane_out)

    lanelines_out = []
    for laneline_in in lanelines_in:
        laneline_out = []
        for pt_3d in laneline_in['pos3DInCameraList']:
            laneline_out.append([pt_3d['x'], pt_3d['y'], pt_3d['z']])
        lanelines_out.append(laneline_out)

    return centerlanes_out, lanelines_out, lane_data['cameraHeight'], lane_data['cameraPitch']


if __name__ == '__main__':
    base_folder = "/home/yuliangguo/Datasets/Apollo_Sim_3D_Lane/"
    input_file = base_folder + "img_list.txt"
    output_gt_file = base_folder + "laneline_label.json"
    image_list, label_list, name_list = get_lists(input_file)
    vis_folder = base_folder + "laneline_vis/"
    if not os.path.exists(vis_folder) and vis:
        os.mkdir(vis_folder)
    # save the full list
    f_out = open(output_gt_file, 'w')
    for i in range(len(image_list)):
        print(i)
        img_vis, valid_img = laneline_label_generator(base_folder, image_list[i], label_list[i], output_gt_file)
        if vis:
            img_name = image_list[i].split('/')
            img_name = img_name[-1]
            print(name_list[i])
            # cv2.imshow('gt visualize', img)
            # cv2.waitKey()
            cv2.imwrite(vis_folder + name_list[i] + '.jpg', img_vis)
    f_out.close()
