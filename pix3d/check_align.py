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

    args = parser.parse_args()

    idx_p3d = int(args.index)
    print('idx_p3d:' , idx_p3d)

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

    RT = np.hstack([np.array(rot_mat),np.array(trans_mat).reshape((-1,1))])
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
    print('RT       : ', RT)
    print('RT[:,:3] : ', RT[:,:3])
    print('RT[:,3:] : ', RT[:,3:])
    xyz = np.dot(xyz, RT[:,:3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:,:2] / xyz[:,2:]
    return xy

if __name__ == '__main__':
    main()