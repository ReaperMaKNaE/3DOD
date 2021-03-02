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
import get_sdf_data
import glob
from mesh_to_sdf import mesh_to_voxels

def main():
    parser = argparse.ArgumentParser(description='OD3D_Reconstruction Process')
    parser.add_argument('--index', default=0, type=int,
                        help='the index of pix3d')
    parser.add_argument('--check_mesh', default=0, type=int,
                        help='True if you want to check individual model')
    parser.add_argument('--model', default='bed',
                        help='select the model when you set check_mesh as True')
    parser.add_argument('--model_index', default=0, type=int,
                        help='the index of the model if you set check_mesh as True')

    args = parser.parse_args()

    idx_p3d = int(args.index)
    print('idx_p3d:' , idx_p3d)
    voxel_size = 0.2

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

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
    rot_mat = json_data[idx_p3d]["rot_mat"]
    trans_mat = json_data[idx_p3d]["trans_mat"]

    RT = np.hstack([np.array(rot_mat), np.array(trans_mat).reshape((-1,1))])

    if args.check_mesh == 1:
        print('load model...')
        mesh = trimesh.load('{}/{}/{}/{}/model.obj'.format(root, model[0], model[1], model[2]))
        print('loading is complete. convert to voxel...')
        voxels = mesh_to_voxels(mesh, 64, pad=True)
        print('Getting voxel is complete.')
        verts, faces, norms, vals = measure.marching_cubes_lewiner(voxels, level=0)
        verts = verts*voxel_size
        vertex_of_obj = np.asarray(verts)
    else :
        mesh = o3d.io.read_triangle_mesh('{}/{}/{}/{}/model.obj'.format(root, model[0], model[1], model[2]))
        vertex_of_obj = np.asarray(mesh.vertices)

    # why side by side mirror-shaped?
    # is this something rule of here?
    K = np.array([[-570. ,0.    ,320.],
                  [0.    ,570.  ,320.],
                  [0.    ,0.    ,1.  ]])

    camera_image = project(vertex_of_obj, K, RT)

    plt.plot(camera_image[:,0], camera_image[:,1])
    plt.show()

def project(xyz, K, RT):
    #print('RT       : ', RT)
    #print('RT[:,:3] : ', RT[:,:3])
    #print('RT[:,3:] : ', RT[:,3:])
    xyz = np.dot(xyz, RT[:,:3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:,:2] / xyz[:,2:]
    return xy

if __name__ == '__main__':
    #get_sdf_data.get_mesh_from_obj('bed', 1)
    main()