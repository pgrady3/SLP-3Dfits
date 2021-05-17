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

SLP_DATASET_PATH = '/home/patrick/datasets/SLP/danaLab'
SLP_CODE_PATH = '/home/patrick/bed/SLP-Dataset-and-Code'
FITS_PATH = './fits'

sys.path.append(SLP_CODE_PATH)   # Set this to the SLP github repo
from data.SLP_RD import SLP_RD


def get_smpl(pkl_data):
    model = smplx.create('models', model_type='smpl', gender=pkl_data['gender'])

    output = model(betas=torch.Tensor(pkl_data['betas']).unsqueeze(0),
                   body_pose=torch.Tensor(pkl_data['body_pose']).unsqueeze(0),
                   transl=torch.Tensor(pkl_data['transl']).unsqueeze(0),
                   global_orient=torch.Tensor(pkl_data['global_orient']).unsqueeze(0),
                   return_verts=True)

    smpl_vertices = output.vertices.detach().cpu().numpy().squeeze()
    smpl_joints = output.joints.detach().cpu().numpy().squeeze()

    smpl_o3d = o3d.geometry.TriangleMesh()
    smpl_o3d.triangles = o3d.utility.Vector3iVector(model.faces)
    smpl_o3d.vertices = o3d.utility.Vector3dVector(smpl_vertices)
    smpl_o3d.compute_vertex_normals()

    smpl_o3d_offset = o3d.geometry.TriangleMesh()
    smpl_o3d_offset.triangles = o3d.utility.Vector3iVector(model.faces)
    smpl_o3d_offset.vertices = o3d.utility.Vector3dVector(smpl_vertices + np.array([1.3, 0, 0]))
    smpl_o3d_offset.compute_vertex_normals()
    smpl_o3d_offset.paint_uniform_color([0.3, 0.4, 0.8])


    all_markers = []
    for i in range(25):
        if np.all(pkl_data['gt_joints'][i, :] == 0):
            continue

        # color = cm.jet((i / 25.0 * 3) % 1)[:3]
        color = [0.0, 0.7, 0.0]

        pos = smpl_joints[util.smpl_to_openpose()[i], :]
        rad = 0.07
        if i == 0:
            ear_joints = util.smpl_to_openpose()[17:19]    # Get left and right ear
            pos = smpl_joints[ear_joints, :].mean(0)
            rad = 0.10

        smpl_marker = util.o3d_get_sphere(color=color, pos=pos, radius=rad)
        # all_markers.append(smpl_marker)

        # z_depth = smpl_joints[util.smpl_to_openpose()[i], 2] - 0.20
        # z_depth = 1.75
        z_depth = smpl_joints[:, 2].min() - 0.05

        gt_pos_3d = util.inverse_camera_perspective_transform(pkl_data['gt_joints'], z_depth,
                                                           camera_rotation=pkl_data['camera_rotation'],
                                                           camera_translation=pkl_data['camera_translation'],
                                                           camera_center=pkl_data['camera_center'],
                                                           camera_focal_length_x=pkl_data['camera_focal_length_x'],
                                                           camera_focal_length_y=pkl_data['camera_focal_length_y'])

        pred_marker = util.o3d_get_sphere(color=color, pos=gt_pos_3d[i, :], radius=0.03)
        all_markers.append(pred_marker)

    return smpl_vertices, model.faces, smpl_o3d, smpl_o3d_offset, all_markers


def get_depth_henry(idx, sample):
    # Get the depth image, but warped to PM
    raw_depth = SLP_dataset.get_array_A2B(idx=idx, modA='depthRaw', modB='depthRaw')

    pointcloud = util.project_depth_with_warping(SLP_dataset, raw_depth, idx)

    valid_z = np.logical_and(pointcloud[:, 2] > 1.65, pointcloud[:, 2] < 2.15)  # Cut out any outliers above the bed
    valid_x = np.logical_and(pointcloud[:, 0] > -0.3, pointcloud[:, 0] < 0.8)  # Cut X
    valid_y = np.logical_and(pointcloud[:, 1] > -1.1, pointcloud[:, 1] < 1.0)  # Cut Y
    valid_all = np.logical_and.reduce((valid_x, valid_y, valid_z))
    pointcloud = pointcloud[valid_all, :]

    ptc_depth = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud))
    return ptc_depth


def get_rgb(idx, sample):
    # Load RGB image
    RGB_to_depth = SLP_dataset.get_array_A2B(idx=idx, modA='RGB', modB='depthRaw')
    depth_raw = np.ones((RGB_to_depth.shape[0], RGB_to_depth.shape[1]), dtype=np.float32) * 2.15
    rows, cols = np.where(RGB_to_depth[:, :, 0] == 0)
    depth_raw[rows, cols] = np.inf

    depth_image = o3d.geometry.Image(depth_raw)
    rgb_image = o3d.geometry.Image(RGB_to_depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image, depth_scale=1)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=RGB_to_depth.shape[1], height=RGB_to_depth.shape[0],
                                                  fx=SLP_dataset.f_d[0], fy=SLP_dataset.f_d[1],
                                                  cx=SLP_dataset.c_d[0], cy=SLP_dataset.c_d[1])

    rgbd_ptc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    rows, cols = np.where(RGB_to_depth[:, :, 0] != 0)
    ptc_colors = np.array(RGB_to_depth[rows, cols, :], dtype=float) / 255.0
    rgbd_ptc.colors = o3d.utility.Vector3dVector(ptc_colors)

    rgbd_ptc = rgbd_ptc.translate((-1.5, 0, 0))
    return rgbd_ptc


def view_fit(sample, idx):
    pkl_path = os.path.join(FITS_PATH, 'p{:03d}'.format(sample[0]), 's{:02d}.pkl'.format(sample[2]))
    pkl_np = pickle.load(open(pkl_path, 'rb'))

    smpl_vertices, smpl_faces, smpl_mesh, smpl_mesh_calc, joint_markers = get_smpl(pkl_np)
    pcd = get_depth_henry(idx, sample)
    rgbd_ptc = get_rgb(idx, sample)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)
    # vis.add_geometry(rgbd_ptc)
    # vis.add_geometry(smpl_mesh)
    vis.add_geometry(smpl_mesh_calc)

    lbl = 'Participant {} sample {}'.format(sample[0], sample[2])
    vis.add_geometry(util.o3d_get_text(lbl, (-0.4, 1.0, 2), direction=(0.01, 0, -1), degree=-90, font_size=150, density=0.2))

    for j in joint_markers:
        vis.add_geometry(j)

    util.o3d_set_camera_extrinsic(vis, np.eye(4))
    # pos = np.eye(4)
    # pos[0, 3] = -1.3
    # util.o3d_set_camera_extrinsic(vis, pos)

    vis.run()
    vis.destroy_window()


def view_dataset(all_samples, skip_sample=0, skip_participant=0):
    for idx, sample in enumerate(all_samples):
        if sample[0] < skip_participant or sample[2] < skip_sample:
            continue

        view_fit(sample, idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sample', type=int, default=0)
    parser.add_argument('-p', '--participant', type=int, default=0)
    args = parser.parse_args()

    class PseudoOpts:
        SLP_fd = SLP_DATASET_PATH
        sz_pch = [256, 256]
        fc_depth = 50
        cov_li = ['uncover']  # give the cover class you want here
    SLP_dataset = SLP_RD(PseudoOpts, phase='all')  # all test result

    view_dataset(SLP_dataset.pthDesc_li, skip_sample=args.sample, skip_participant=args.participant)
