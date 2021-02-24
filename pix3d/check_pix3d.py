# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
import random
from IPython.display import display, Image
from PIL import Image

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

from pix3dLoader import Detectron2_FasterRCNN_Pix3D
from detectron2 import model_zoo

root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
classes_123 = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
for d in ['train', 'val', 'test']:
    DatasetCatalog.register('/content/output/pix3d_20210222_{}'.format(d), lambda d=d: Detectron2_FasterRCNN_Pix3D(root, '{}.txt'.format(d)))
    MetadataCatalog.get('/content/output/pix3d_20210222_{}'.format(d)).set(thing_classes=classes_123)

pix3d_metadata = MetadataCatalog.get("/content/output/pix3d_20210222_val")

print('pix3d_metadata:', pix3d_metadata)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = '/content/output/model_final.pth'
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set a custom testing threshold
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.002
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = Detectron2_FasterRCNN_Pix3D(root, 'test.txt')

index = 0
for d in random.sample(dataset_dicts, 3):   
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=pix3d_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    print('outputs:', outputs["instances"].to("cpu"))
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('/content/output/output_{}.jpg'.format(index),out.get_image()[:, :, ::-1])
    index += 1