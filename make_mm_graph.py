import sys
import numpy as np
import open3d as o3d
import os
import torch
import pickle
import smplx
import argparse
import matplotlib.cm as cm
import util
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

joint_titles = ['Head', 'Neck', # 0, 1
                'R Shoulder', 'R Elbow', 'R Wrist', # 2, 3, 4
                'L Shoulder', 'L Elbow', 'L Wrist', # 5, 6, 7
                'R Hip', 'R Knee', 'R Ankle',   # 8, 9, 10
                'L Hip', 'L Knee', 'L Ankle']   # 11, 12, 13

show_order = [1, 2, 5, 3, 6, 4, 7, 8, 11, 9, 12, 10, 13]

joint_combos = {'Neck': [1], 'Sholder': [2, 5], 'Elbow': [3, 6], 'Wrist': [5, 7], 'Hip': [8, 11], 'Knee': [9, 12], 'Ankle': [10, 13],
                'Average': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}

if __name__ == "__main__":
    all_joint_err = pickle.load(open('joint_error.pkl', 'rb'))

    # print(all_joint_err)
    total_error = []
    for joint in range(len(all_joint_err[0])):
        errors = []
        for p in all_joint_err:
            errors.append(np.linalg.norm(p[joint]['gt'] - p[joint]['smpl']))

        errors = np.array(errors)
        print('Errors {} {:.4f}'.format(joint, errors.mean()))

        # throwout_joints = [0, 2, 5, 8, 11]    # Head, shoulders, hips
        # throwout_joints = [0]   # Head
        # if joint not in throwout_joints:
        total_error.append(errors.mean())

    total_error = np.array(total_error)
    print('all errors', total_error)
    print('whole dataset error', total_error[1:].mean())

    # bar_lbls = [joint_titles[i] for i in show_order]
    # bar_vals = [total_error[i] for i in show_order]
    #
    # plt.bar(bar_lbls, bar_vals)
    # plt.show()


    bar_lbls = [i for i in joint_combos.keys()]
    bar_vals = [total_error[i].mean() * 1000 for i in joint_combos.values()]

    plt.style.use('ggplot')
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    # Then, "ALWAYS use sans-serif fonts"
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rcParams['figure.figsize'] = [4.5, 3.5]
    plt.xticks(rotation='vertical')
    barlist = plt.bar(bar_lbls, bar_vals, color='darkblue')
    barlist[-1].set_color('darkred')

    plt.ylabel('Joint error (mm)')
    plt.title('SLP-3Dfits Annotation vs Fit Joint Error')

    plt.grid(True, linestyle='--')
    plt.subplots_adjust(bottom=0.2)

    plt.show()

