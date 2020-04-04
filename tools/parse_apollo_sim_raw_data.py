#!/usr/bin/env python
"""
Process laneline raw labels for the synthetic 3D lane dataset, and save in .json file.

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""
import os
import cv2
import numpy as np
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

"""
Segmentation background class color coding: R G B

Untagged 0 0 0
LaneMarking 255 255 255
Road 171 99 36
Sidewalk 169 94 30
SpeedBump 192 134 83
GuardRail 174 92 20
Terrain 114 75 41
Building 113 67 173
TrafficLight 118 67 185
Pole 104 71 147
StreetLight 73 50 103
GarageMarker 60 27 103
TrafficSign 101 24 201
Vegetation 0 204 20
Sky 0 197 204
TrafficCone 42 92 110
Barricade 205 205 205
"""
# color coding of different lane labels
bg_colors = [[0, 0, 0], [255, 255, 255], [171, 99, 36], [169, 94, 30], [192, 134, 83],
             [174, 92, 20], [114, 75, 41], [113, 67, 173], [118, 67, 185], [104, 71, 147],
             [73, 50, 103], [60, 27, 103], [101, 24, 201], [0, 204, 20], [0, 197, 204],
             [42, 92, 110], [205, 205, 205]]

img_height = 1080
img_width = 1920
K = np.array([[2015.0,      0, 960.0],
             [      0, 2015.0, 540.0],
             [      0,      0,     1]])
merge = True
parse_visibility = True
vis_dist_th = 2


def get_lists(test_file):
    with open(test_file) as f:
        test_list = f.readlines()

    image_list = []
    label_list = []
    seg_list = []
    depth_list = []
    name_list = []
    for line in test_list:
        line = line.replace("\n", "")
        line = line.replace("./", "")
        image_list.append("images/" + line)
        label_list.append("labels/" + line[:-4] + ".txt")
        seg_list.append("segmentation/" + line[:-4] + ".png")
        depth_list.append("depth/" + line[:-4] + ".png")
        name_list.append(line[:-4].replace("/", "_"))
    return image_list, label_list, seg_list, depth_list, name_list


def merge_segments_recursive(centerlane, centerline_dict, laneline_dict, centerline2del, laneline2del):
    centerlane['successorList'] = [sid for sid in centerlane['successorList'] if sid in centerline_dict]
    # handle 1 to 1 and 2 to 1 cases by extending the first, and marking the second segment to delete
    # Recursively merge and update successorList of the first segment
    if len(centerlane['successorList']) == 1:
        centerlane2 = centerline_dict[centerlane['successorList'][0]]
        new_succssorList = merge_segments_recursive(centerlane2, centerline_dict, laneline_dict, centerline2del, laneline2del)
        centerlane['successorList'] = new_succssorList

        # this condition could be removed, but kept for safe
        if -0.01 <= centerlane2['pos3DInCameraList'][0]['z'] - centerlane['pos3DInCameraList'][-1]['z'] < 0.01:
            centerlane['pos3DInCameraList'].extend(centerlane2['pos3DInCameraList'])
            centerline2del[centerlane2['id']] = 1

        # merge associated lanelines
        if centerlane['leftBoundaryId'] in laneline_dict and centerlane2['leftBoundaryId'] in laneline_dict:
            left_laneline = laneline_dict[centerlane['leftBoundaryId']]
            left_laneline2 = laneline_dict[centerlane2['leftBoundaryId']]
            # only merge those have not been dealt from other centerlane associations
            if -0.01 <= left_laneline2['pos3DInCameraList'][0]['z'] - left_laneline['pos3DInCameraList'][-1]['z'] < 0.01:
                left_laneline['pos3DInCameraList'].extend(left_laneline2['pos3DInCameraList'])
                laneline2del[left_laneline2['id']] = 1

        if centerlane['rightBoundaryId'] in laneline_dict and centerlane2['rightBoundaryId'] in laneline_dict:
            right_laneline = laneline_dict[centerlane['rightBoundaryId']]
            right_laneline2 = laneline_dict[centerlane2['rightBoundaryId']]
            # only merge those have not been dealt from other centerlane associations
            if -0.01 <= right_laneline2['pos3DInCameraList'][0]['z'] - right_laneline['pos3DInCameraList'][-1]['z'] < 0.01:
                right_laneline['pos3DInCameraList'].extend(right_laneline2['pos3DInCameraList'])
                laneline2del[right_laneline2['id']] = 1

        return centerlane['successorList']
    else:
        return centerlane['successorList']


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

    centerlines_in = lane_data['laneList']
    lanelines_in = lane_data['laneBoundaryList']

    # register each lane by its id for access
    centerline_dict = {lane['id']: lane for lane in centerlines_in}
    centerline2del = {lane['id']: 0 for lane in centerlines_in}

    laneline_dict = {lane['id']: lane for lane in lanelines_in}
    laneline2del = {lane['id']: 0 for lane in lanelines_in}

    if merge:
        """
        This merging algorithms serves the specific purpose of the 3D LaneNet representation where the sharing portion
        are duplicated into two lanes when dealing with many-to-one and one-to-many connectivity.
        
        The algorithm is a two-iteration solution. The first iteration solves all the one-to-one connections. A 
        recursive function is applied to keep merging second segment with its successors, delete second segment's id 
        from first segment's successor list, and add the merged successor lane's successor id to the list. The second 
        iteration deals will one-to-many connection case. The first segment will be marked and to delete, and the second
        segment is augmented with the first segment at front.
        """

        # iter1: merge centerlines based on successorList: all the modification applies directly back to centerlines_in and lanelines_in
        for id, centerlane in centerline_dict.items():
            merge_segments_recursive(centerlane, centerline_dict, laneline_dict, centerline2del, laneline2del)

        # iter2: handle 1 to 2 case by extending the second and marking the first segment to delete
        for id, centerlane in centerline_dict.items():
            if len(centerlane['successorList']) > 1:
                for second_id in centerlane['successorList']:
                    centerlane2 = centerline_dict[second_id]
                    # this condition could be removed, but kept for safe
                    if -0.01 <= centerlane2['pos3DInCameraList'][0]['z'] - centerlane['pos3DInCameraList'][-1]['z'] < 0.01:
                        centerlane2['pos3DInCameraList'] = centerlane['pos3DInCameraList'] + centerlane2['pos3DInCameraList']
                        centerline2del[id] = 1

                    # merge associated lanelines
                    if centerlane['leftBoundaryId'] in laneline_dict and centerlane2['leftBoundaryId'] in laneline_dict:
                        left_laneline = laneline_dict[centerlane['leftBoundaryId']]
                        left_laneline2 = laneline_dict[centerlane2['leftBoundaryId']]
                        # only merge those have not been dealt from other centerlane associations
                        if -0.01 <= left_laneline2['pos3DInCameraList'][0]['z'] - left_laneline['pos3DInCameraList'][-1]['z'] < 0.01:
                            left_laneline2['pos3DInCameraList'] = left_laneline['pos3DInCameraList'] + left_laneline2['pos3DInCameraList']
                            laneline2del[left_laneline['id']] = 1

                    if centerlane['rightBoundaryId'] in laneline_dict and centerlane2['rightBoundaryId'] in laneline_dict:
                        right_laneline = laneline_dict[centerlane['rightBoundaryId']]
                        right_laneline2 = laneline_dict[centerlane2['rightBoundaryId']]
                        # only merge those have not been dealt from other centerlane associations
                        if -0.01 <= right_laneline2['pos3DInCameraList'][0]['z'] - right_laneline['pos3DInCameraList'][-1]['z'] < 0.01:
                            right_laneline2['pos3DInCameraList'] = right_laneline['pos3DInCameraList'] + right_laneline2['pos3DInCameraList']
                            laneline2del[right_laneline['id']] = 1

    # convert to output format
    centerlanes_out = []
    for i, centerlane_in in enumerate(centerlines_in):
        if centerline2del[centerlane_in['id']] or centerlane_in['type'] == 'SHOULDER':
            # add its inner side associated laneline into delete list
            if centerlane_in['pos3DInCameraList'][0]['x'] < 0 and centerlane_in['rightBoundaryId'] in laneline_dict:
                laneline2del[centerlane_in['rightBoundaryId']] = 1
            elif centerlane_in['pos3DInCameraList'][0]['x'] > 0 and centerlane_in['leftBoundaryId'] in laneline_dict:
                laneline2del[centerlane_in['leftBoundaryId']] = 1
            continue
        centerlane_out = []
        for pt_3d in centerlane_in['pos3DInCameraList']:
            centerlane_out.append([pt_3d['x'], pt_3d['y'], pt_3d['z']])
        centerlanes_out.append(centerlane_out)

    lanelines_out = []
    for i, laneline_in in enumerate(lanelines_in):
        if laneline2del[laneline_in['id']]:
            continue
        laneline_out = []
        for pt_3d in laneline_in['pos3DInCameraList']:
            laneline_out.append([pt_3d['x'], pt_3d['y'], pt_3d['z']])
        lanelines_out.append(laneline_out)

    return centerlanes_out, lanelines_out, lane_data['cameraHeight'], lane_data['cameraPitch']


def laneline_label_generator(base_folder, image_file, label_file, seg_file, depth_file, output_gt_file):
    img = cv2.imread(base_folder + image_file)
    seg_img = cv2.imread(base_folder + seg_file)
    depth_img = cv2.imread(base_folder + depth_file).astype(np.float32)/255
    depth_label_map = (depth_img[:, :, 2] + depth_img[:, :, 1]/255)*655.36
    # label_img = np.zeros((img_height, img_width), dtype=np.int8)

    # if image_name == 'images/00/0000062.jpg':
    #     print('here')

    # extract lanes and types from ground-truth label file
    centerlines, lanelines, cam_height, cam_pitch = process_lane_label_apollo_sim_3D(base_folder + label_file)

    # compute projection matrix
    proj_g2c = np.array([[1,                             0,                              0,          0],
                         [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                         [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0],
                         [0,                             0,                              0,          1]])
    proj_c2g = np.linalg.inv(proj_g2c)
    centerlines_g = []
    lanelines_g = []
    centerlines_visibility = []
    lanelines_visibility = []

    # convert laneline to ground coordinate and visualize therm by projecting to image
    for i, lane3D in enumerate(centerlines):
        lane3D = np.array(lane3D)
        # project laneline 3D in camera coordinates to ground coordinates
        centerline = np.concatenate([lane3D, np.ones([lane3D.shape[0], 1])], axis=1)
        centerline_g = np.matmul(centerline, proj_c2g.T)
        # centerline_g = centerline_g[centerline_g[:, 2] > 0.01, :]
        centerlines_g.append(centerline_g[:, :3].tolist())

        # project to image (camera coordinates to image coordinates)
        lane2D = np.matmul(lane3D, K.T)
        lane2D = np.divide(lane2D, np.expand_dims(lane2D[:, 2], -1))

        # determine visibility
        visibility_vec = np.ones(lane2D.shape[0])
        for j in range(lane2D.shape[0]):
            # decide from image border range
            if lane2D[j, 0] < 1 or lane2D[j, 0] >= img_width-1:
                visibility_vec[j] = 0
                continue
            if lane2D[j, 1] < 1 or lane2D[j, 1] >= img_height-1:
                visibility_vec[j] = 0
                continue
            # decide from orientation: the portion close to horizontal will be considered as invisible
            if j > 0 and (
                    abs(lane2D[j - 1, 0] - lane2D[j, 0]) / (abs(lane2D[j - 1, 1] - lane2D[j, 1]) + 0.000001) > 15 or
                    lane2D[j - 1, 1] - lane2D[j, 1] <= 0):
                visibility_vec[j] = 0

        lane2D = lane2D.astype(np.int)
        for j in range(lane2D.shape[0]):
            if visibility_vec[j] == 0:
                continue
            # decide from depth map
            class_color = [seg_img[lane2D[j, 1], lane2D[j, 0], 2],
                           seg_img[lane2D[j, 1], lane2D[j, 0], 1],
                           seg_img[lane2D[j, 1], lane2D[j, 0], 0]]
            if np.abs(depth_label_map[lane2D[j, 1], lane2D[j, 0]] - centerline[j][2]) > vis_dist_th and \
                    np.abs(depth_label_map[lane2D[j, 1] + 1, lane2D[j, 0]] - centerline[j][2]) > vis_dist_th and \
                    np.abs(depth_label_map[lane2D[j, 1] - 1, lane2D[j, 0]] - centerline[j][2]) > vis_dist_th and\
                    np.abs(depth_label_map[lane2D[j, 1], lane2D[j, 0] + 1] - centerline[j][2]) > vis_dist_th and\
                    np.abs(depth_label_map[lane2D[j, 1], lane2D[j, 0] - 1] - centerline[j][2]) > vis_dist_th and \
                    np.abs(depth_label_map[lane2D[j, 1] + 1, lane2D[j, 0] + 1] - centerline[j][2]) > vis_dist_th and \
                    np.abs(depth_label_map[lane2D[j, 1] - 1, lane2D[j, 0] + 1] - centerline[j][2]) > vis_dist_th and \
                    np.abs(depth_label_map[lane2D[j, 1] + 1, lane2D[j, 0] - 1] - centerline[j][2]) > vis_dist_th and \
                    np.abs(depth_label_map[lane2D[j, 1] - 1, lane2D[j, 0] - 1] - centerline[j][2]) > vis_dist_th and \
                    class_color in bg_colors:
                visibility_vec[j] = 0
                continue

        # visibility_vec = medfilt(visibility_vec, 5)
        # find the first and last visible index, assume all points in between visible
        visible_indices = np.where(visibility_vec > 0)[0]
        if visible_indices.shape[0] > 0:
            # vis_start = visible_indices[0]
            vis_start = 0
            vis_end = visible_indices[-1]
            visibility_vec[vis_start:vis_end] = 1
        centerlines_visibility.append(visibility_vec.tolist())

        # determine visibility and draw on image
        for j in range(1, lane2D.shape[0]):
            if visibility_vec[j] > 0:
                img = cv2.line(img,
                               (lane2D[j-1, 0], lane2D[j-1, 1]),
                               (lane2D[j, 0], lane2D[j, 1]),
                               color=[255, 0, 0], thickness=3)
            else:
                img = cv2.line(img,
                               (lane2D[j - 1, 0], lane2D[j - 1, 1]),
                               (lane2D[j, 0], lane2D[j, 1]),
                               color=[0, 0, 0], thickness=3)
        # draw end points
        cv2.circle(img, (int(lane2D[0, 0]), int(lane2D[0, 1])), 3, [0, 0, 255], 2)
        cv2.circle(img, (int(lane2D[-1, 0]), int(lane2D[-1, 1])), 3, [0, 0, 255], 2)

    for i, lane3D in enumerate(lanelines):
        lane3D = np.array(lane3D)
        # project laneline 3D in camera coordinates to ground coordinates
        laneline = np.concatenate([lane3D, np.ones([lane3D.shape[0], 1])], axis=1)
        laneline_g = np.matmul(laneline, proj_c2g.T)
        # laneline_g = laneline_g[laneline_g[:, 2] > 0.01, :]
        lanelines_g.append(laneline_g[:, :3].tolist())
        # project to image (camera coordinates to image coordinates)
        lane2D = np.matmul(np.array(lane3D), K.T)
        lane2D = np.divide(lane2D, np.expand_dims(lane2D[:, 2], -1))

        # determine visibility
        visibility_vec = np.ones(lane2D.shape[0])
        for j in range(lane2D.shape[0]):
            # decide from image border range
            if lane2D[j, 0] < 1 or lane2D[j, 0] >= img_width-1:
                visibility_vec[j] = 0
                continue
            if lane2D[j, 1] < 1 or lane2D[j, 1] >= img_height-1:
                visibility_vec[j] = 0
                continue
            # decide from orientation: the portion close to horizontal will be considered as invisible
            if j > 0 and (
                    abs(lane2D[j - 1, 0] - lane2D[j, 0]) / (abs(lane2D[j - 1, 1] - lane2D[j, 1]) + 0.000001) > 15 or
                    lane2D[j - 1, 1] - lane2D[j, 1] <= 0):
                visibility_vec[j] = 0

        lane2D = lane2D.astype(np.int)
        for j in range(lane2D.shape[0]):
            if visibility_vec[j] == 0:
                continue

            class_color = [seg_img[lane2D[j, 1], lane2D[j, 0], 2],
                           seg_img[lane2D[j, 1], lane2D[j, 0], 1],
                           seg_img[lane2D[j, 1], lane2D[j, 0], 0]]
            if np.abs(depth_label_map[lane2D[j, 1], lane2D[j, 0]] - laneline[j][2]) > vis_dist_th and \
                    np.abs(depth_label_map[lane2D[j, 1] + 1, lane2D[j, 0]] - laneline[j][2]) > vis_dist_th and \
                    np.abs(depth_label_map[lane2D[j, 1] - 1, lane2D[j, 0]] - laneline[j][2]) > vis_dist_th and\
                    np.abs(depth_label_map[lane2D[j, 1], lane2D[j, 0] + 1] - laneline[j][2]) > vis_dist_th and\
                    np.abs(depth_label_map[lane2D[j, 1], lane2D[j, 0] - 1] - laneline[j][2]) > vis_dist_th and \
                    np.abs(depth_label_map[lane2D[j, 1] + 1, lane2D[j, 0] + 1] - laneline[j][2]) > vis_dist_th and \
                    np.abs(depth_label_map[lane2D[j, 1] - 1, lane2D[j, 0] + 1] - laneline[j][2]) > vis_dist_th and \
                    np.abs(depth_label_map[lane2D[j, 1] + 1, lane2D[j, 0] - 1] - laneline[j][2]) > vis_dist_th and \
                    np.abs(depth_label_map[lane2D[j, 1] - 1, lane2D[j, 0] - 1] - laneline[j][2]) > vis_dist_th and \
                    class_color in bg_colors:
                visibility_vec[j] = 0
                continue

        # visibility_vec = medfilt(visibility_vec, 5)
        # find the first and last visible index, assume all points in between visible
        visible_indices = np.where(visibility_vec > 0)[0]
        if visible_indices.shape[0] > 0:
            # vis_start = visible_indices[0]
            vis_start = 0
            vis_end = visible_indices[-1]
            visibility_vec[vis_start:vis_end] = 1
        lanelines_visibility.append(visibility_vec.tolist())

        # draw on image
        for j in range(1, lane2D.shape[0]):
            if visibility_vec[j] > 0:
                img = cv2.line(img,
                               (lane2D[j-1, 0], lane2D[j-1, 1]),
                               (lane2D[j, 0], lane2D[j, 1]),
                               color=[0, 255, 0], thickness=3)
            else:
                img = cv2.line(img,
                               (lane2D[j - 1, 0], lane2D[j - 1, 1]),
                               (lane2D[j, 0], lane2D[j, 1]),
                               color=[0, 0, 0], thickness=3)
        # draw end points
        cv2.circle(img, (int(lane2D[0, 0]), int(lane2D[0, 1])), 3, [0, 0, 255], 2)
        cv2.circle(img, (int(lane2D[-1, 0]), int(lane2D[-1, 1])), 3, [0, 0, 255], 2)

    # generate json
    result = {}
    valid_img = True
    result["raw_file"] = image_file
    result['cam_height'] = cam_height
    result['cam_pitch'] = cam_pitch
    result["centerLines"] = np.asarray(centerlines_g).tolist()
    result["laneLines"] = np.asarray(lanelines_g).tolist()
    result["centerLines_visibility"] = np.asarray(centerlines_visibility).tolist()
    result["laneLines_visibility"] = np.asarray(lanelines_visibility).tolist()

    with open(output_gt_file, "a") as outfile:
        outfile.write(json.dumps(result))
        outfile.write("\n")
        outfile.close()

    return img, valid_img


if __name__ == '__main__':
    vis = False
    base_folder = "/media/yuliangguo/DATA1/Datasets/Apollo_Sim_3D_Lane_release/"
    input_file = base_folder + "img_list.txt"
    output_gt_file = base_folder + "laneline_label.json"
    vis_folder = base_folder + "laneline_vis/"
    if not os.path.exists(vis_folder) and vis:
        os.mkdir(vis_folder)

    # get lists of file names
    image_list, label_list, seg_list, depth_list, name_list = get_lists(input_file)

    # generate label associated with each image
    f_out = open(output_gt_file, 'w')
    for i in range(len(image_list)):
        print(i)
        img_vis, valid_img = laneline_label_generator(base_folder, image_list[i], label_list[i],
                                                      seg_list[i], depth_list[i], output_gt_file)
        if vis:
            img_name = image_list[i].split('/')
            img_name = img_name[-1]
            print(name_list[i])
            # cv2.imshow('gt visualize', img)
            # cv2.waitKey()
            cv2.imwrite(vis_folder + name_list[i] + '.jpg', img_vis)
    f_out.close()
