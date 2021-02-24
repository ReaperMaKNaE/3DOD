from mesh_to_sdf import mesh_to_voxels

import trimesh
import skimage.measure
import os
import numpy as np
import glob

root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def get_sdf_from_obj(root):
    """ get_sdf_from_obj make SDF(Signed Distance Function) from .obj file

    category_list     : list of categories
    model_list        : list of models(generally, IKEA) for each category
    """
    model_list = []
    category_list = os.listdir('{}/model/'.format(root))
    print('category_list: ', category_list, ' is checked')
    for i in category_list:
        model_list.append(os.listdir('{}/model/{}/'.format(root, i)))

    index = 1
    total_num = 0
    for i in model_list:
        total_num += len(i)
    print('Total number of the models: ', total_num)

    for idx, comp in enumerate(category_list):
        print('category: ', comp)
        print('model list for this category: ', model_list[idx])
        for j in model_list[idx]:
            object_file = glob.glob('{}/model/{}/{}/*.obj'.format(root,comp,j))
            mesh = trimesh.load(object_file[0])
            voxels = mesh_to_voxels(mesh, 64, pad=True)
            
            data = {'tsdf': voxels}
            np.savez_compressed('{}/model/{}/{}/tsdf.npz'.format(root, comp, j), **data)
            print('{}/model/{}/{} is transformed to SDF completely. [{}/{}]'.format(root, comp, j, index, total_num))
            index += 1

def check_3D_data(root):
    # there are too many models in some cases, but i think all of them are same.
    # it can be checked below code.
    # Therefore, i will use only first taken-model even though there are many models in same folder.
    root = os.path.join("{}/model/chair".format(root))
    model_list = os.listdir(root)
    root_list = []
    for i in model_list:
        if len(glob.glob('{}/{}/*.obj'.format(root,i))) != 1:
            print('Model path: {}/{}'.format(root,i), 'the number of models: ', len(glob.glob('{}/{}/*.obj'.format(root,i))))
            root_list.append('{}/{}'.format(root,i))

    model_list = []
    # below code make the terminal too complicated
    for _, comp in enumerate(root_list):
        model_list.append(glob.glob('{}/*.obj'.format(comp)))
    
    print('model_list[0][0]: ', model_list[0][0])
    print('model_list[0][1]: ', model_list[0][1])
    

#vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(voxels, level=0)
#mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
#mesh.show()

get_sdf_from_obj(root)
#check_3D_data(root)