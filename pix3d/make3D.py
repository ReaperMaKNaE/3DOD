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

def main():
    parser = argparse.ArgumentParser(description='OD3D_Reconstruction Process')
    parser.add_argument('--index', default=0, type=int,
                        help='the index of pix3d')

    args = parser.parse_args()
    idx_p3d = int(args.index)
    print('idx_p3d:' , idx_p3d)

    voxel_size = 0.4
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    room_size = [256, 256, 256]
    wall_thickness = 5

    with open("{}/pix3d.json".format(root)) as json_file:
        json_data = json.load(json_file)

    img = json_data[idx_p3d]["img"]
    print('img root: {}/{}'.format(root,img))
    img = cv2.imread('{}/{}'.format(root,img), cv2.IMREAD_COLOR)
    cv2.imshow('images', img)
    cv2.waitKey(0)
    print('model: ', json_data[idx_p3d]["model"])
    model = json_data[idx_p3d]["model"].split('/')
    rot_mat = json_data[idx_p3d]["rot_mat"]
    # make room
    room = np.ones([room_size[0],room_size[1],room_size[2]])
    # make floor
    #room[:,:,:wall_thickness] = np.zeros([room_size[0],room_size[1],wall_thickness])
    # make wall
    #room[:wall_thickness,:,:] = np.zeros([wall_thickness,room_size[1],room_size[2]])
    #room[:,:wall_thickness,:] = np.zeros([room_size[0],wall_thickness,room_size[2]])
    room.astype(float)

    # make objects
    objects = []

    object1_path = os.path.join(root, model[0], model[1], model[2], 'tsdf.npz')
    object1 = call_tsdf(object1_path)

    #object2 = '{}/model/chair/IKEA_BERNHARD/tsdf.npz'.format(root)
    #object2 = call_tsdf(object2)

    scale_factor = [0.3, 0.6]
    object_scale = [[int(room_size[0]*scale_factor[0]), int(room_size[1]*scale_factor[0]), int(room_size[2]*scale_factor[0])],
                     [int(room_size[0]*scale_factor[1]), int(room_size[1]*scale_factor[1]), int(room_size[2]*scale_factor[1])]]
    center_offset = [[(room_size[0]-object_scale[0][0])/2, (room_size[1]-object_scale[0][1])/2, (room_size[2]-object_scale[0][2])/2],
                     [(room_size[0]-object_scale[1][0])/2, (room_size[1]-object_scale[1][1])/2, (room_size[2]-object_scale[1][2])/2]]
    center_offset_x = [int(center_offset[0][0]), int(center_offset[1][0])]
    center_offset_y = [int(center_offset[0][1]), int(center_offset[1][1])]
    center_offset_z = [int(center_offset[0][2]), int(center_offset[1][2])]

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
                     
    #object2 = np.pad(object2, ((center_offset_x[1], room_size[0]-center_offset_x[1]-object2.shape[0]),
    #                           (center_offset_y[1], room_size[1]-center_offset_y[1]-object2.shape[1]),
    #                           (center_offset_z[1], room_size[2]-center_offset_z[1]-object2.shape[2])),
    #                 mode = 'constant', constant_values=1)

    # rotate and localization module.
    # if you want watch your object in this section, plz refer below conditions
    # angle \in [0, 2\pi]
    # offset \in [-1, 1]

    

    rotation_matrices=[]
    rotation_matrices.append(rot_mat)

    object1 = rotate_object(object1, rotation_mat=rotation_matrices[0], angle=[-20,180,-12], offset=[0.3, 0.0, 0.0])
    #object2 = rotate_object(object2, angle=[0,0,0], offset=[0.0, 0.0, 0.0])

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
    o3d.visualization.draw_geometries([pcd, mesh])

def call_tsdf(filename):
    tsdf = np.load(filename)
    return tsdf['tsdf']

def get_mesh(tsdf, voxel_size):
    verts, faces, norms, colors = measure.marching_cubes_lewiner(tsdf, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts*voxel_size
    return verts, faces, norms, colors

def rotate_object(object_item, rotation_mat=None, angle=[0,0,0], offset=[0,0,0]):
    object1_torch = torch.from_numpy(object_item)
    for i in range(2):
        object1_torch = object1_torch.unsqueeze(0)

    roll  = angle[0] * np.pi / 180.
    yaw   = angle[1] * np.pi / 180.
    pitch = angle[2] * np.pi / 180.

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

    if rotation_mat:
        rot_mat = np.dot(rotation_mat, rot_mat)

    rot_mat_torch = torch.from_numpy(rot_mat).float()

    rotated_object = localization(object1_torch, rot_mat_torch, offset)
    rotated_object = rotated_object.numpy()

    return rotated_object

def localization(input_tensor, rotation_matrix, offset=[0,0,0]):
    device_ = input_tensor.device
    _, _, d, h, w = input_tensor.shape
    locations_3d = get_3d_locations(d, h, w, device_)
    rotated_3d_positions = torch.bmm(rotation_matrix.view(1,3,3).expand(d*h*w, 3, 3), locations_3d).view(1, d, h, w, 3)
    rot_locs = torch.split(rotated_3d_positions, split_size_or_sections=1, dim=4)
    # print('rot_locs shapes: ', rot_locs[0].shape, rot_locs[1].shape, rot_locs[2].shape)
    normalised_locs_x = (2.0*rot_locs[0]+offset[0]*(w-1))/(w-1)#- (w-1))/(w-1)
    normalised_locs_y = (2.0*rot_locs[1]+offset[1]*(h-1))/(h-1)# - (h-1))/(h-1)
    normalised_locs_z = (2.0*rot_locs[2]+offset[2]*(d-1))/(d-1)# - (d-1))/(d-1)
    grid = torch.stack([normalised_locs_x, normalised_locs_y, normalised_locs_z], dim=4).view(1, d, h, w, 3)
    rotated_signal = F.grid_sample(input=input_tensor, grid=grid, padding_mode="border", mode='nearest', align_corners=True)
    return rotated_signal.squeeze(0).squeeze(0)

def get_3d_locations(d, h, w, device_):
    locations_x = torch.linspace(-w/2, w/2, w).view(1, 1, 1, w).to(device_).expand(1, d, h, w)
    locations_y = torch.linspace(-h/2, h/2, h).view(1, 1, h, 1).to(device_).expand(1, d, h, w)
    locations_z = torch.linspace(-d/2, d/2, d).view(1, d, 1, 1).to(device_).expand(1, d, h, w)
    locations_3d = torch.stack([locations_x, locations_y, locations_z], dim=4).view(-1, 3, 1)
    return locations_3d

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