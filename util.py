import numpy as np
import open3d as o3d
import sys
from pyquaternion import Quaternion
from PIL import Image, ImageFont, ImageDraw


SLP_CODE_PATH = '/home/patrick/bed/SLP-Dataset-and-Code'
sys.path.append(SLP_CODE_PATH)
import utils.utils as ut    # SLP utils


def inverse_camera_perspective_transform(points, z_dist,
                                         camera_rotation, camera_translation, camera_center,
                                         camera_focal_length_x, camera_focal_length_y):

    camera_mat = np.zeros((2, 2))   # Make a 2x2 matrix with focal length on diagonal
    camera_mat[0, 0] = camera_focal_length_x
    camera_mat[1, 1] = camera_focal_length_y
    camera_mat_inv = np.linalg.inv(camera_mat)

    camera_transform = np.eye(4)    # Make 4x4 rigid transform matrix
    camera_transform[:3, :3] = camera_rotation
    camera_transform[:3, 3] = camera_translation
    camera_transform_inv = np.linalg.inv(camera_transform)

    pixel_points = np.matmul(camera_mat_inv, points.T - camera_center[..., np.newaxis]).T
    img_points = np.ones((points.shape[0], 4))
    img_points[:, :2] = pixel_points * z_dist
    img_points[:, 2] = z_dist

    projected_points = np.matmul(camera_transform_inv, img_points.T).T  # Apply rigid transform
    non_homo_points = projected_points[:, :3]

    return non_homo_points


def smpl_to_openpose():
    """
    :return: Returns the indices of the permutation that maps OpenPose joints to SMPL joints
    """

    return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                     7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                     dtype=np.int32)


def o3d_get_text(text, pos, direction=None, degree=-90.0, density=10, font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', font_size=10):
    """
    Generate a Open3D text point cloud used for visualization.
    https://github.com/intel-isl/Open3D/issues/2
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    font_obj = ImageFont.truetype(font, int(font_size * density))
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 1000 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def o3d_get_sphere(color=[0.3, 1.0, 0.3], pos=[0, 0, 0], radius=0.06):
    """
    Generates an Open3D mesh object as a sphere
    :param color: RGB color 3-list, 0-1
    :param pos: Center of sphere, 3-list
    :param radius: Radius of sphere
    :return: Open3D TriangleMesh sphere
    """
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=5)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color(color)
    mean = np.asarray(mesh_sphere.vertices).mean(axis=0)
    diff = np.asarray(pos) - mean
    mesh_sphere.translate(diff)
    return mesh_sphere


def o3d_set_camera_extrinsic(vis, transform=np.eye(4)):
    """
    Sets the Open3D camera position and orientation
    :param vis: Open3D visualizer object
    :param transform: 4x4 numpy defining a rigid transform where the camera should go
    """
    ctr = vis.get_view_control()
    cam = ctr.convert_to_pinhole_camera_parameters()
    cam.extrinsic = transform
    ctr.convert_from_pinhole_camera_parameters(cam)


def apply_homography(points, h, yx=True):
    # Apply 3x3 homography matrix to points
    # Note that the homography matrix is parameterized as XY,
    # but all image coordinates are YX

    if yx:
        points = np.flip(points, 1)

    points_h = np.concatenate((points, np.ones((points.shape[0], 1))), 1)
    tform_h = np.matmul(h, points_h.T).T
    tform_h = tform_h / tform_h[:, 2][:, np.newaxis]

    points = tform_h[:, :2]

    if yx:
        points = np.flip(points, 1)

    return points


def get_modified_depth_to_pressure_homography(slp_dataset, idx):
    """
    Magic function to get the homography matrix to warp points in depth-cloud space into pressure-mat space.
    However, this modifies the scaling of the homography matrix to keep the same scale, so that the points can be
    projected into 3D metric space.
    :param slp_dataset: Dataset object to extract homography from
    :param idx: SLP dataset sample index
    :return:
    """
    WARPING_MAGIC_SCALE_FACTOR = (192. / 345.)  # Scale matrix to align to PM. 192 is height of pressure mat, 345 is height of bed in depth pixels

    depth_Tr = slp_dataset.get_PTr_A2B(idx=idx, modA='depthRaw', modB='PM')     # Get SLP homography matrix
    depth_Tr /= depth_Tr[2, 2]  # Make more readable matrix

    depth_Tr[0:2, 0:3] = depth_Tr[0:2, 0:3] / WARPING_MAGIC_SCALE_FACTOR
    return depth_Tr


def project_depth_with_warping(slp_dataset, depth_arr, idx):
    """
    Project a 2D depth image into 3D space. Additionally, this warps the 2D points of the input image
    using a homography matrix. Importantly, there is no interpolation in the warping step.
    :param slp_dataset: Dataset object to extract image from
    :param depth_arr: The input depth image to use
    :param idx: SLP dataset sample index
    :return: A [N, 3] numpy array of the 3D pointcloud
    """
    # The other methods of using cv2.warpPerspective apply interpolation to the image, which is bad. This doesn't
    # Input image is YX
    depth_homography = get_modified_depth_to_pressure_homography(slp_dataset, idx)
    orig_x, orig_y = np.meshgrid(np.arange(0, depth_arr.shape[1]), np.arange(0, depth_arr.shape[0]))

    image_space_coordinates = np.stack((orig_x.flatten(), orig_y.flatten()), 0).T
    warped_image_space_coordinates = apply_homography(image_space_coordinates, depth_homography, yx=False)

    cd_modified = np.matmul(depth_homography, np.array([slp_dataset.c_d[0], slp_dataset.c_d[1], 1.0]).T)    # Multiply the center of the depth image by homography
    cd_modified = cd_modified/cd_modified[2]    # Re-normalize

    projection_input = np.concatenate((warped_image_space_coordinates, depth_arr.flatten()[..., np.newaxis]), 1)
    ptc = ut.pixel2cam(projection_input, slp_dataset.f_d, cd_modified[0:2]) / 1000.0
    return ptc
