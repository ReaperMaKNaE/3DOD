import numpy as np
import os
from skimage import measure
import open3d as o3d
from scipy.interpolate import RegularGridInterpolator
import torch
import torch.nn.functional as F
import math

def main():
    voxel_size = 0.4
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    # make room
    room = np.ones([128,128,128])
    # make floor
    room[:,:,:5] = np.zeros([128,128,5])
    # make wall
    room[:5,:,:] = np.zeros([5,128,128])
    room[:,:5,:] = np.zeros([128,5,128])
    room.astype(float)

    offset = 15
    # make objects
    objects = []
    object1 = '{}/model/bed/IKEA_BEDDINGE/tsdf.npz'.format(root)
    object1 = call_tsdf(object1)

    scale_factor = 0.5
    object1_scale = (int(128*scale_factor),int(128*scale_factor),int(128*scale_factor))

    # rescale object using torch.nn.functional.interpolate
    object1_torch = torch.from_numpy(object1)
    for i in range(2):
        object1_torch = object1_torch.unsqueeze(0)
    object1_torch = F.interpolate(object1_torch, object1_scale, mode='trilinear')
    for i in range(2):
        object1_torch = object1_torch.squeeze(0)
    object1 = object1_torch.numpy()

    # this part is padding 1(which means air)
    # but this is not a best option. we should change this part to torch.nn.functional.pad
    # actually, this process is similar to grid_sample
    object1 = np.pad(object1, ((offset, 128-offset-object1.shape[0]),
                               (offset, 128-offset-object1.shape[1]),
                               (offset, 128-offset-object1.shape[2])),
                     mode = 'constant', constant_values=1)
    object1 = rotate_object(object1, angle= 10)
    #object1 = rotate_object(object1, rot_tran_mat)
    objects.append(object1)

    # resize and localization for objects
    for idx, comp in enumerate(objects):
        room += objects[idx]
    room -= len(objects)

    #print(object1.shape)
    #object1 = get_mesh(object1, voxel_size)
    #pcwrite('{}/pc.ply'.format(root), object1)
    #check_ply('{}/pc.ply'.format(root))
    
    room = get_mesh(room, voxel_size)

    pcwrite('{}/pc.ply'.format(root), room)
    check_ply('{}/pc.ply'.format(root))

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

def check_ply(filename):
    pcd = o3d.io.read_point_cloud(filename)
    # I can't see anything within there...
    #pcd.paint_uniform_color([0.5,0.5,0.5])
    o3d.visualization.draw_geometries([pcd])

def call_tsdf(filename):
    tsdf = np.load(filename)
    return tsdf['tsdf']

def get_mesh(tsdf, voxel_size):
    verts = measure.marching_cubes_lewiner(tsdf, level=0)[0]
    verts_ind = np.round(verts).astype(int)
    verts = verts*voxel_size
    return verts

def rotate_object(object_item, angle=0):
    object1_torch = torch.from_numpy(object_item)
    for i in range(2):
        object1_torch = object1_torch.unsqueeze(0)

    roll  = angle * np.pi / 180.
    yaw   = angle * np.pi / 180.
    pitch = angle * np.pi / 180.

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

    return rotated_object

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

def rotate(input_tensor, rotation_matrix):
    device_ = input_tensor.device
    _, _, d, h, w = input_tensor.shape
    locations_3d = get_3d_locations(d, h, w, device_)
    rotated_3d_positions = torch.bmm(rotation_matrix.view(1,3,3).expand(d*h*w, 3, 3), locations_3d).view(1, d, h, w, 3)
    rot_locs = torch.split(rotated_3d_positions, split_size_or_sections=1, dim=4)
    normalised_locs_x = (2.0*rot_locs[0] - (w-1))/(w-1)
    normalised_locs_y = (2.0*rot_locs[1] - (h-1))/(h-1)
    normalised_locs_z = (2.0*rot_locs[2] - (d-1))/(d-1)
    grid = torch.stack([normalised_locs_x, normalised_locs_y, normalised_locs_z], dim=4).view(1, d, h, w, 3)
    rotated_signal = F.grid_sample(input=input_tensor, grid=grid, mode='nearest', align_corners=True)
    return rotated_signal.squeeze(0).squeeze(0)

def get_3d_locations(d, h, w, device_):
    locations_x = torch.linspace(0, w-1, w).view(1, 1, 1, w).to(device_).expand(1, d, h, w)
    locations_y = torch.linspace(0, h-1, h).view(1, 1, h, 1).to(device_).expand(1, d, h, w)
    locations_z = torch.linspace(0, d-1, d).view(1, d, 1, 1).to(device_).expand(1, d, h, w)
    locations_3d = torch.stack([locations_x, locations_y, locations_z], dim=4).view(-1, 3, 1)
    return locations_3d
    

def trilinear_interpolation(vectors, coord):
    """ actually, not trilinear interpolation now.
    this is nearest neighbor
    but this is abandonned cuz too heavy
    """
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

    return rotated_coord

if __name__=='__main__':
    main()
    #rotation_check()