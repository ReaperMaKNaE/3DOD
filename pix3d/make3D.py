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

    rot_tran_mat = np.array([1,0,0,0],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1])

    # make room
    room = np.ones([128,128,128])
    # make floor
    room[:,:,:5] = np.zeros([128,128,5])
    # make wall
    room[:5,:,:] = np.zeros([5,128,128])
    room[:,:5,:] = np.zeros([128,5,128])
    room.astype(float)

    offset = 5
    # make objects
    objects = []
    object1 = '{}/model/bed/IKEA_BEDDINGE/tsdf.npz'.format(root)
    object1 = call_tsdf(object1)

    scale_factor = 0.7
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
    object1 = np.pad(object1, ((offset, 128-offset-object1.shape[0]),
                               (offset, 128-offset-object1.shape[1]),
                               (offset, 128-offset-object1.shape[2])),
                     mode = 'constant', constant_values=1)
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

#def rotate_object(object_item, matrix):

def rotation_check():
    

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
    print(rot_mat)

if __name__=='__main__':
    #main()
    rotation_check()