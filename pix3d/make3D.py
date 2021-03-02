import numpy as np
import os
from skimage import measure
import open3d as o3d
from scipy.interpolate import RegularGridInterpolator
import torch
import torch.nn.functional as F
import math
import trimesh
import json
import argparse
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def main():
    parser = argparse.ArgumentParser(description='OD3D_Reconstruction Process')
    parser.add_argument('--index', default=0, type=int,
                        help='the index of pix3d')
    parser.add_argument('--set_angles', default=0, type=float,
                        help='set the angles')

    args = parser.parse_args()
    idx_p3d = int(args.index)
    print('idx_p3d:' , idx_p3d)

    voxel_size = 0.4
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    scale_factor = [0.5, 0.5]
    room_scale = 0.8
    room_size = [256, 256, 256]
    #angles = [-20,185,-12]
    angles = [0,0,0]
    room_scale = (int(room_size[0]*room_scale), int(room_size[1]*room_scale), int(room_size[2]*room_scale))
    wall_thickness = 5

    with open("{}/pix3d.json".format(root)) as json_file:
        json_data = json.load(json_file)

    img = json_data[idx_p3d]["img"]
    print('img root: {}/{}'.format(root,img))
    img = cv2.imread('{}/{}'.format(root,img), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640,640))
    cv2.imshow('images', img)
    cv2.waitKey(0)
    print('model: ', json_data[idx_p3d]["model"])
    model = json_data[idx_p3d]["model"].split('/')
    #basic_camera_location = json_data[0]["cam_position"]
    #basic_trans_mat = json_data[0]["trans_mat"]
    #basic_trans_mat[2] -= 1
    rot_mat = json_data[idx_p3d]["rot_mat"]
    trans_mat = json_data[idx_p3d]["trans_mat"]

    RT = np.hstack([np.array(rot_mat),np.array(trans_mat).reshape((-1,1))])

    #trans_mat = -np.dot(np.linalg.inv(np.array(rot_mat)),np.array(trans_mat))

    print('trans_mat: ', trans_mat)

    #angles[0] += np.arctan(trans_mat[1]/trans_mat[0])*args.set_angles*180/np.pi
    #angles[1] += np.arctan(trans_mat[2]/trans_mat[1])*args.set_angles*180/np.pi
    #angles[2] += np.arctan(trans_mat[0]/trans_mat[2])*args.set_angles*180/np.pi

    print('angles: ', angles)
    """     cam_position = json_data[idx_p3d]["cam_position"]

    print('cam_position: ', cam_position)

    calculated_pose = np.dot(np.array(rot_mat), np.array(trans_mat))
    print('calculated_pose: ', calculated_pose) """

    # FUCKFUCKFUCKFUCKFUCK
    """    camera_location = json_data[idx_p3d]["cam_position"]
    trans_mat = json_data[idx_p3d]["trans_mat"]
    trans_mat[2] -= 1

    basic_campose_trans = np.subtract(np.array(basic_camera_location), np.array(basic_trans_mat))
    campose_trans = np.subtract(np.array(camera_location), np.array(trans_mat))

    basic_campose_trans = basic_campose_trans.reshape((-1,1))
    basic_campose_trans = np.hstack([np.array([[0],[1],[0]]), basic_campose_trans])
    basic_campose_trans = np.hstack([np.array([[1],[0],[0]]), basic_campose_trans])

    campose_trans = campose_trans.reshape((-1,1))
    campose_trans = np.hstack([np.array([[rot_mat[0][1]],[rot_mat[1][1]],[rot_mat[2][1]]]), campose_trans])
    campose_trans = np.hstack([np.array([[rot_mat[0][0]],[rot_mat[1][0]],[rot_mat[2][0]]]), campose_trans])

    basic_campose_trans_inv = np.linalg.inv(basic_campose_trans)
    rotation_mat_campose_trans = np.dot(campose_trans, basic_campose_trans_inv)
    print('rot_mat: ', rot_mat)
    print('campose_trans: ', campose_trans)
    print('rotation_mat_campose_trans: ', rotation_mat_campose_trans) """

    # make room
    room = np.ones([room_size[0],room_size[1],room_size[2]])
    # make floor, z-axis
    room[:,:,room_size[2]-wall_thickness:room_size[2]] = np.zeros([room_size[0],room_size[1],wall_thickness])
    # make wall
    # x-axis
    room[room_size[0]-wall_thickness:room_size[0],:,:] = np.zeros([wall_thickness,room_size[1],room_size[2]])
    # y-axis
    room[:,:wall_thickness,:] = np.zeros([room_size[0],wall_thickness,room_size[2]])
    room.astype(float)

    # make objects
    objects = []

    object1_path = os.path.join(root, model[0], model[1], model[2], 'tsdf.npz')
    object1 = call_tsdf(object1_path)

    #mesh = o3d.io.read_triangle_mesh('{}/{}/{}/{}/model.obj'.format(root, model[0], model[1], model[2]))
    #vertex_of_obj = np.asarray(mesh.vertices)
    #o3d.visualization.draw_geometries([mesh])

    #object2 = '{}/model/chair/IKEA_BERNHARD/tsdf.npz'.format(root)
    #object2 = call_tsdf(object2)

    room_offset = [(room_size[0]-room.shape[0])/2, (room_size[1]-room.shape[1])/2, (room_size[2]-room.shape[2])/2]
    room_offset_x = int(room_offset[0])
    room_offset_y = int(room_offset[1])
    room_offset_z = int(room_offset[2])

    object_scale = [[int(room_size[0]*scale_factor[0]), int(room_size[1]*scale_factor[0]), int(room_size[2]*scale_factor[0])],
                     [int(room_size[0]*scale_factor[1]), int(room_size[1]*scale_factor[1]), int(room_size[2]*scale_factor[1])]]
    center_offset = [[(room_size[0]-object_scale[0][0])/2, (room_size[1]-object_scale[0][1])/2, (room_size[2]-object_scale[0][2])/2],
                     [(room_size[0]-object_scale[1][0])/2, (room_size[1]-object_scale[1][1])/2, (room_size[2]-object_scale[1][2])/2]]
    center_offset_x = [int(center_offset[0][0]), int(center_offset[1][0])]
    center_offset_y = [int(center_offset[0][1]), int(center_offset[1][1])]
    center_offset_z = [int(center_offset[0][2]), int(center_offset[1][2])]

    room_torch = torch.from_numpy(room)
    for i in range(2):
        room_torch = room_torch.unsqueeze(0)
    room_torch = F.interpolate(room_torch, room_scale, mode='trilinear')
    for i in range(2):
        room_torch = room_torch.squeeze(0)
    room = room_torch.numpy()

    # rescale object using torch.nn.functional.interpolate
    object1_torch = torch.from_numpy(object1)
    for i in range(2):
        object1_torch = object1_torch.unsqueeze(0)
    object1_torch = F.interpolate(object1_torch, object_scale[0], mode='trilinear')
    for i in range(2):
        object1_torch = object1_torch.squeeze(0)
    object1 = object1_torch.numpy()

    #object2_torch = torch.from_numpy(object2)
    #for i in range(2):
    #    object2_torch = object2_torch.unsqueeze(0)
    #object2_torch = F.interpolate(object2_torch, object_scale[1], mode='trilinear')
    #for i in range(2):
    #    object2_torch = object2_torch.squeeze(0)
    #object2 = object2_torch.numpy()

    # this part is padding 1(which means air)
    # but this is not a best option. we should change this part to torch.nn.functional.pad
    # actually, this process is similar to grid_sample
    object1 = np.pad(object1, ((center_offset_x[0], room_size[0]-center_offset_x[0]-object1.shape[0]),
                               (center_offset_y[0], room_size[1]-center_offset_y[0]-object1.shape[1]),
                               (center_offset_z[0], room_size[2]-center_offset_z[0]-object1.shape[2])),
                     mode = 'constant', constant_values=1)
                     
    room = np.pad(room, ((room_offset_x, room_size[0]-room_offset_x-room.shape[0]),
                         (room_offset_y, room_size[1]-room_offset_y-room.shape[1]),
                         (room_offset_z, room_size[2]-room_offset_z-room.shape[2])),
                  mode='constant', constant_values=1)

    #object2 = np.pad(object2, ((center_offset_x[1], room_size[0]-center_offset_x[1]-object2.shape[0]),
    #                           (center_offset_y[1], room_size[1]-center_offset_y[1]-object2.shape[1]),
    #                           (center_offset_z[1], room_size[2]-center_offset_z[1]-object2.shape[2])),
    #                 mode = 'constant', constant_values=1)

    # rotate and localization module.
    # if you want watch your object in this section, plz refer below conditions
    # angle \in [0, 2\pi]
    # offset \in [-1, 1]

    
    # angles = [-12, 185, -20]

    rotation_matrices=[]
    rotation_matrices.append(rot_mat)

    # Align viewpoint
    # To align, 1. rotate -90 at y-axis
    #           2. rotate -90 at x-axis
    # offset means similiar meaning of translation
    room = rotate_object(room, rotation_mat=rotation_matrices[0], offset=[0.05, -0.8, 0.15])
    object1 = rotate_object(object1, basic_angle=[[0,0,0],[0,0,0]], rotation_mat=rotation_matrices[0])

    objects.append(object1)
    #objects.append(object2)

    # resize and localization for objects
    room += objects[0]
    room -= 1
    #for idx, comp in enumerate(objects):
    #    room += objects[idx]
    #room -= len(objects)

    #print(object1.shape)
    #object1 = get_mesh(object1, voxel_size)
    #pcwrite('{}/pc.ply'.format(root), object1)
    #check_ply('{}/pc.ply'.format(root))
    
    verts, faces, norms, _ = get_mesh(room, voxel_size)
    
    pcwrite('{}/pc.ply'.format(root), verts)
    meshwrite('{}/mesh.ply'.format(root), verts, faces, norms)
    check_mesh_and_ply(root)

    #check_ply('{}/pc.ply'.format(root))

    #check_mesh('{}/mesh.ply'.format(root))

def pcwrite(filename, xyz):
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("end_header\n")

    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f\n"%(
            xyz[i,0], xyz[i,1], xyz[i,2],
        ))
    ply_file.close()

def meshwrite(filename, verts, faces, norms):
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("element face %d\n"%(faces.shape[0]))
    def apply_camera(vis):
        intrinsicVal=o3d.camera.PinholeCameraIntrinsic()
        intrinsicVal.set_intrinsics(w, h, f, f, w/2.0-0.5, h/2.0-0.5)
        
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f\n"%(
            verts[i,0], verts[i,1], verts[i,2],
            norms[i,0], norms[i,1], norms[i,2],
        ))
    
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n"%(faces[i,0], faces[i,1], faces[i,2]))

    ply_file.close()

def check_ply(filename):
    pcd = o3d.io.read_point_cloud(filename)
    # I can't see anything within there...
    #pcd.paint_uniform_color([0.5,0.5,0.5])
    o3d.visualization.draw_geometries([pcd])

def check_mesh(filename):
    mesh = trimesh.load(filename)
    mesh.show()

def check_mesh_and_ply(root):
    pcd = o3d.io.read_point_cloud('{}/pc.ply'.format(root))
    mesh = o3d.io.read_triangle_mesh('{}/mesh.ply'.format(root))

    w=640
    h=480
    f=570.0

    def capture_depth(vis):
        depth=vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    key_to_callback = {}
    key_to_callback[ord(",")] = capture_depth

    o3d.visualization.draw_geometries_with_key_callbacks([pcd, mesh], key_to_callback)

def call_tsdf(filename):
    tsdf = np.load(filename)
    return tsdf['tsdf']

def get_mesh(tsdf, voxel_size):
    verts, faces, norms, colors = measure.marching_cubes_lewiner(tsdf, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts*voxel_size
    return verts, faces, norms, colors

def rotate_object(object_item, basic_angle = [[0,0,0]], rotation_mat=None, rotation_campose_trans=None, angle=[[0,0,0]], offset=[0,0,0]):
    """ Rotation and translation module
       Args:
       object_item            : object which i want to rotate
       basic_angle            : rotation angle for align         [x, y, z]
       rotation_mat           : rotation matrix in pix3d.json
       rotation_campose_trans : calculated rotation matrix from cam position and trans_mat in pix3d.json
       angle                  : additional angle input           [x, y, z]
       offset                 : additional translation input

       output                 : rotated object(numpy)
    """
    object1_torch = torch.from_numpy(object_item).float()
    for i in range(2):
        object1_torch = object1_torch.unsqueeze(0)

    for idx, comp in enumerate(basic_angle):
        basic_rot_mat0 = get_rotation_matrix(comp)
        if idx == 0:
            basic_rot_mat = basic_rot_mat0
        else:
            basic_rot_mat = np.dot(basic_rot_mat0, basic_rot_mat)
    for idx, comp in enumerate(angle):
        additional_rot_mat0 = get_rotation_matrix(comp)
        if idx == 0:
            additional_rot_mat = additional_rot_mat0
        else:
            additional_rot_mat = np.dot(additional_rot_mat0, additional_rot_mat)

    rot_mat = np.dot(additional_rot_mat, basic_rot_mat)

    if rotation_mat:
        rot_mat = np.dot(rotation_mat, rot_mat)

    if rotation_campose_trans:
        rot_mat = np.dot(rot_mat, rotation_campose_trans)

    rot_mat_torch = torch.from_numpy(rot_mat).float()

    rotated_object = localization(object1_torch, rot_mat_torch, offset)
    rotated_object = rotated_object.numpy()

    return rotated_object

def localization(input_tensor, rotation_matrix, offset=[0,0,0]):
    """ Localization module
       input_tensor    : object which i want to rotate
       rotation_matrix : rotation matrix which will apply to input object
       offset          : same as translation matrix, but not -Rt, only t.

       output          : rotated object(Tensor)
    """
    print('rotation matrix in localization mat : ', rotation_matrix)
    device_ = input_tensor.device
    #_, _, w, h, d = input_tensor.shape
    _, _, d, h, w = input_tensor.shape
    locations_3d = get_3d_locations(d, h, w, device_)
    rotated_3d_positions = torch.bmm(rotation_matrix.view(1,3,3).expand(d*h*w, 3, 3), locations_3d).view(1, d, h, w, 3)
    rot_locs = torch.split(rotated_3d_positions, split_size_or_sections=1, dim=4)
    # print('rot_locs shapes: ', rot_locs[0].shape, rot_locs[1].shape, rot_locs[2].shape)
    normalised_locs_x = (2.0*rot_locs[0]+offset[0]*(w-1))/(w-1)
    normalised_locs_y = (2.0*rot_locs[1]+offset[1]*(h-1))/(h-1)
    normalised_locs_z = (2.0*rot_locs[2]+offset[2]*(d-1))/(d-1)
    grid = torch.stack([normalised_locs_x, normalised_locs_y, normalised_locs_z], dim=4).view(1, d, h, w, 3)
    print('the type of each arguments: ', input_tensor.dtype, grid.dtype)
    rotated_signal = F.grid_sample(input=input_tensor, grid=grid, padding_mode="border", mode='nearest', align_corners=True)
    return rotated_signal.squeeze(0).squeeze(0)

def get_3d_locations(d, h, w, device_):
    """ Get Coordinate of object
       d       : depth(which is z value) of object
       h       : height(which is y value) of object
       w       : width(which is x value) of object

       output  : 3d Coordinate 
    """
    locations_x = torch.linspace(-w/2, w/2, w).view(1, 1, 1, w).to(device_).expand(1, d, h, w)
    locations_y = torch.linspace(-h/2, h/2, h).view(1, 1, h, 1).to(device_).expand(1, d, h, w)
    locations_z = torch.linspace(-d/2, d/2, d).view(1, d, 1, 1).to(device_).expand(1, d, h, w)
    locations_3d = torch.stack([locations_x, locations_y, locations_z], dim=4).view(-1, 3, 1)
    return locations_3d

def get_rotation_matrix(angle=[0,0,0]):
    """ get rotation matrix
       To align axis, change roll and pitch (z-axis rotation for first index, x-axis rotation for third index)
    """
    roll  = angle[0] * np.pi / 180. # roll  : z-axis rotation
    yaw   = angle[1] * np.pi / 180. # yaw   : y-axis rotation
    pitch = angle[2] * np.pi / 180. # pitch : x-axis rotation

    # z-axis rotation
    rot_roll   = np.array([[math.cos(roll), -math.sin(roll), 0.],
                        [math.sin(roll), math.cos(roll),  0.],
                        [0.,             0.,              1.]])
    # y-axis rotation
    rot_yaw    = np.array([[math.cos(yaw),  0., math.sin(yaw)],
                        [0.,             1.,            0.],
                        [-math.sin(yaw), 0., math.cos(yaw)]])
    # x-axis rotation
    rot_pitch  = np.array([[1.,              0.,               0.],
                        [0., math.cos(pitch), -math.sin(pitch)],
                        [0., math.sin(pitch),  math.cos(pitch)]])

    return np.dot(rot_roll, np.dot(rot_yaw, rot_pitch))

def project(xyz, K, RT):
    print('RT       : ', RT)
    print('RT[:,:3] : ', RT[:,:3])
    print('RT[:,3:] : ', RT[:,3:])
    xyz = np.dot(xyz, RT[:,:3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:,:2] / xyz[:,2:]
    return xy

""" 
### USELESS PART
def rotation_check():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    object1 = '{}/model/bed/IKEA_BEDDINGE/tsdf.npz'.format(root)
    object1 = call_tsdf(object1)
    offset = 5
    voxel_size = 0.4
    object1 = np.pad(object1, ((offset, 128-offset-object1.shape[0]),
                               (offset, 128-offset-object1.shape[1]),
                               (offset, 128-offset-object1.shape[2])),
                     mode = 'constant', constant_values=1)

    object1_torch = torch.from_numpy(object1)
    for i in range(2):
        object1_torch = object1_torch.unsqueeze(0)
    #center_vector = [object1.shape[0]/2, object1.shape[1]/2, object1.shape[2]/2]
    
    #coord_vectors = []
    #for i in range(object1.shape[0]):
    #    for j in range(object1.shape[1]):
    #        for k in range(object1.shape[2]):
    #            coord_vectors.append([i-center_vector[0], j-center_vector[1], k-center_vector[2]])

    #transformed_vectors = []

    roll  = 10 * np.pi / 180.
    yaw   = 10 * np.pi / 180.
    pitch = 10 * np.pi / 180.

    rot_roll   = np.array([[math.cos(roll), -math.sin(roll), 0.],
                           [math.sin(roll), math.cos(roll),  0.],
                           [0.,             0.,              1.]])
    rot_yaw    = np.array([[math.cos(yaw),  0., math.sin(yaw)],
                           [0.,             1.,            0.],
                           [-math.sin(yaw), 0., math.cos(yaw)]])
    rot_pitch  = np.array([[1.,              0.,               0.],
                           [0., math.cos(pitch), -math.sin(pitch)],
                           [0., math.sin(pitch),  math.cos(pitch)]])

    rot_mat = np.dot(rot_roll, np.dot(rot_yaw, rot_pitch))

    rot_mat_torch = torch.from_numpy(rot_mat).float()

    rotated_object = rotate(object1_torch, rot_mat_torch)
    rotated_object = rotated_object.numpy()

    room = get_mesh(rotated_object, voxel_size)

    pcwrite('{}/pc.ply'.format(root), room)
    check_ply('{}/pc.ply'.format(root))

    #for i in range(len(coord_vectors)):
    #    transformed_vectors.append(np.add(np.dot(rot_mat, coord_vectors[i]),center_vector))
    
    #print('transformed_vectors: ', transformed_vectors)
    #print('the number of transformed_vectors: ', len(transformed_vectors))

    #rotated_object1 = trilinear_interpolation(transformed_vectors, object1)

def trilinear_interpolation(vectors, coord):
    rotated_coord = coord

    x_range = coord.shape[0]
    y_range = coord.shape[1]
    z_range = coord.shape[2]

    for i in range(x_range):
        for j in range(y_range):
            for k in range(z_range):
                points = []
                for idx in range(len(vectors)):
                    points.append(np.sum(np.abs(np.subtract(vectors[idx],np.array([i,j,k])))))
                points = np.array(points)
                closest_point_idx = np.argmin(points)
                closest_point_x = math.floor(closest_point_idx/(x_range*y_range))
                closest_point_y = math.floor((closest_point_idx - closest_point_x*x_range*y_range)/y_range)
                closest_point_z = closest_point_idx - closest_point_x*x_range*y_range - closest_point_y * y_range
                rotated_coord[i,j,k] = coord[closest_point_x, closest_point_y, closest_point_z]
                print('{}/{} completed'.format(i*x_range*y_range + j*y_range + k, len(vectors)))

    return rotated_coord """

if __name__=='__main__':
    main()
    #rotation_check()