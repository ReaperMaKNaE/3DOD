# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

import random

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import os

from pix3dLoader import Detectron2_FasterRCNN_Pix3D
from detectron2 import model_zoo

def detectron2_Faster_RCNN(args, root):
    print('the datatype and pretrained will be : |||', args.dataset_type, '|||', args.pretrained, '|||')
    print('if pretrained is True, then the model is pretrained faster RCNN at COCO dataset')
    classes = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    for d in ['train', 'test']:
        DatasetCatalog.register('pix3d_{}'.format(d), lambda d=d: Detectron2_FasterRCNN_Pix3D(root, '{}.txt'.format(d)))
        MetadataCatalog.get('pix3d_{}'.format(d)).set(thing_classes=classes)


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.SOLVER.MAX_ITER = 1500    # 300 iterations seems good enough, but you can certainly train longer
    if args.pretrained == True:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        pix3d_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    else:
        cfg.MODEL.WEIGHTS = '{}/weight/model_final_epoch_3000_transform.pth'.format(root)
        #cfg.MODEL.WEIGHTS = '{}/weight/model_final.pth'.format(root)
        pix3d_metadata = MetadataCatalog.get('pix3d_train')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_2D   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    #print('metadatacatalog:', MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))

    from detectron2.utils.visualizer import ColorMode
    
    if args.dataset_type == 'pix3d':
        # Test with test files of Pix3D
        dataset_dicts = Detectron2_FasterRCNN_Pix3D(root, 'test.txt')
        index = 0

        output_dict = []

        for d in random.sample(dataset_dicts, 3):   
            objects = []
            pred_class = []

            im = cv2.imread(d["file_name"])
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                        metadata=pix3d_metadata, 
                        scale=1.0, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            #print('outputs:', outputs["instances"].to("cpu"))
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            output_img = cv2.resize(out.get_image()[:, :, ::-1], dsize=(640,640), interpolation=cv2.INTER_AREA)
            #cv2.imwrite('/content/output/output_{}.jpg'.format(index),out.get_image()[:, :, ::-1])
            #cv2.imshow('output_{}'.format(index), out.get_image()[:, :, ::-1])
            cv2.imshow('output_{}'.format(index), output_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            index += 1

            bbox = outputs["instances"].to("cpu").pred_boxes.tensor.cpu().tolist()
            pred_classes = outputs["instances"].to("cpu").pred_classes.tolist()
            if bbox:
                for i in range(len(bbox)):
                    objects.append(im[int(bbox[i][1]):int(bbox[i][3]),
                                    int(bbox[i][0]):int(bbox[i][2])])
                    pred_class.append(classes[pred_classes[i]])

            output_dict.append([objects, pred_class])
        
    elif args.dataset_type == 'SUNRGBD':
        root = os.path.join(root, '..', 'SUNRGBD/crop')
        image_list = os.listdir(root)
        index = 0

        output_dict = []

        for d in random.sample(image_list, 3):
            d = '{}/{}'.format(root, d)
            print('d:', d)   
            objects = []
            pred_class = []

            im = cv2.imread(d)
            outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            v = Visualizer(im[:, :, ::-1],
                        #MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                        metadata=pix3d_metadata, 
                        scale=1.0, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
            #print('outputs:', outputs["instances"].to("cpu"))
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            output_img = cv2.resize(out.get_image()[:, :, ::-1], dsize=(640,640), interpolation=cv2.INTER_AREA)
            #cv2.imwrite('/content/output/output_{}.jpg'.format(index),out.get_image()[:, :, ::-1])
            #cv2.imshow('output_{}'.format(index), out.get_image()[:, :, ::-1])
            cv2.imshow('output_{}'.format(index), output_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            index += 1

            bbox = outputs["instances"].to("cpu").pred_boxes.tensor.cpu().tolist()
            pred_classes = outputs["instances"].to("cpu").pred_classes.tolist()
            if bbox:
                for i in range(len(bbox)):
                    objects.append(im[int(bbox[i][1]):int(bbox[i][3]),
                                      int(bbox[i][0]):int(bbox[i][2])])
                    pred_class.append(classes[pred_classes[i]])

            output_dict.append([objects, pred_class])

    return output_dict