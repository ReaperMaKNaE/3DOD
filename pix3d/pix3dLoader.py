import torch.utils.data as data
import numpy as np
from path import Path
import random
import cv2
import json
from detectron2.structures import BoxMode
import itertools
import random
import os

def load_as_float(path):
    return cv2.imread(path).astype(np.float32)

def Detectron2_FasterRCNN_Pix3D(root, ttype):
    classes = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']

    # load ttype file
    with open("{}/{}".format(root, ttype), 'r') as f:
        data = f.readlines()
        data_strip = [strip.rstrip() for strip in data]

    # load json
    with open("{}/pix3d.json".format(root)) as json_file:
        json_data = json.load(json_file)
    dataset_dicts = []
    # get data
    for _, comp in enumerate(json_data):
        if comp["img"] in data_strip:
            record = {}
            img = comp["img"]
            img_size = comp["img_size"]
            category = comp["category"]
            model = comp["model"]
            _2d_keypoints = comp["2d_keypoints"]
            _3d_keypoints = comp["3d_keypoints"]
            bbox = comp["bbox"]
            rot_mat = comp["rot_mat"]
            trans_mat = comp["trans_mat"]
            cam_position = comp["cam_position"]
            focal_length = comp["focal_length"]
            inplane_rotation = comp["inplane_rotation"]

            record['file_name'] = '{}/{}'.format(root,img)
            #print('record, filename: ', record["file_name"])
            record["image_id"] = category
            record["height"] = img_size[1]
            record["width"] = img_size[0]
            # test matrix
            record["rot_mat"] = rot_mat
            record["trans_mat"] = trans_mat
            #
            #print('image_height: ', img_size[1], 'image_width:', img_size[0])
            #print('category, idx:', category, classes.index(category))
            obj = {
                'bbox': bbox,
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': classes.index(category)
            }
            record["annotations"] = [obj]
            dataset_dicts.append(record)

    return dataset_dicts

class Pix3DLoader(data.Dataset):
    """Pix3DLoader load file as written at each type: train, val, test
       pix3d/img/bed/0001.png
	   pix3d/img/bed/0002.png
	   ...
	   pix3d/img/wardrobe/0001.jpg
	   
	   the extensions of the file: jpg, jpeg, png
	"""

    def __init__(self, root, seed=None, ttype='train.txt', transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/ttype
        self.ttype = ttype

        # load ttype file
        with open("{}/{}".format(self.root, self.ttype), 'r') as f:
            data = f.readlines()
            data_strip = [strip.rstrip() for strip in data]

        self.data_p3d = sorted(data_strip)
        #self.transform = transform

    def __getitem__(self, index):
        # load json
        with open("{}/pix3d.json".format(self.root)) as json_file:
            json_data = json.load(json_file)

        # get data
        for idx, comp in enumerate(json_data):
            if comp["img"] == self.data_p3d[index]:
                idx_p3d = idx
                continue

        img = json_data[idx_p3d]["img"]
        category = json_data[idx_p3d]["category"]
        model = json_data[idx_p3d]["model"]
        _2d_keypoints = json_data[idx_p3d]["2d_keypoints"]
        _3d_keypoints = json_data[idx_p3d]["3d_keypoints"]
        bbox = json_data[idx_p3d]["bbox"]
        rot_mat = json_data[idx_p3d]["rot_mat"]
        trans_mat = json_data[idx_p3d]["trans_mat"]
        cam_position = json_data[idx_p3d]["cam_position"]
        focal_length = json_data[idx_p3d]["focal_length"]
        inplane_rotation = json_data[idx_p3d]["inplane_rotation"]

        img = cv2.imread('{}/{}'.format(root, img))

        return img, category, model, _2d_keypoints, _3d_keypoints, bbox, rot_mat, trans_mat, cam_position, focal_length, inplane_rotation

    def __len__(self):
        return len(self.data_p3d)

root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

data_dict = Detectron2_FasterRCNN_Pix3D(root, ttype='train.txt')

rot_mat = np.array(data_dict[0]["rot_mat"])
trans_mat = np.array(data_dict[0]["trans_mat"])

_RT = np.dot(-rot_mat, trans_mat)
proj_mat = np.hstack([rot_mat, _RT[:,None]])
proj_mat = np.vstack([proj_mat, np.array([0,0,0,1])])
print(proj_mat)