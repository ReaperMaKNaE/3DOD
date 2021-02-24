import torch
import numpy as np
import argparse
import os
import cv2

import detectron2_FasterRCNN
import pix3dLoader

from pix3dLoader import Pix3DLoader

root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# the list for valuable to make argparser:
# epochs, epoch-size, lr, batch-size, weight-decay, print-frequency,
# seed, ttype, training-output-frequency and so on.

classes = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']

parser = argparse.ArgumentParser(description='OD3D_Reconstruction Process')
parser.add_argument('--dataset', default='{}/dataset/'.format(root), metavar='STR',
                    help='the location of dataset')
parser.add_argument('--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--dataset_type', default='pix3d', metavar='STR',
                    help='pix3d or SUNRGBD')
parser.add_argument('--pretrained', default=False,
                    help='if pretrained yes, the pretrained network will take.')
parser.add_argument('--score_2D', default = 0.7, type=float,
                    help='the threshold score for 2D object detection')

args = parser.parse_args()
#Pix3DDataset = Pix3DLoader(

#)

#t

def output_detectron2():
    Object_detector_2D = detectron2_FasterRCNN.detectron2_Faster_RCNN

    # the output format is
    #
    #                       1st image                         2nd image
    # output = [ [[object images], [categories]], [[object images], [categories]], ..... ]
    #
    # But generally, i will use only 1 image input for scene.

    output = Object_detector_2D(args, root)

    # for data check
    for idx, comp in enumerate(output):
        if comp[0]:
            print('the number of detected comp: ', len(comp[0]))
            for i, img in enumerate(comp[0]):
                img = cv2.resize(img, (640,640), interpolation=cv2.INTER_AREA)
                cv2.imshow("cropped_image_{}".format(i), img)
                cv2.imwrite("{}/cropped_image_{}_{}.jpg".format(root,idx,i), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            #cv2.imshow("cropped image", comp[0][0])
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        else:
            print('nothing detected at image {}'.format(idx))


output_detectron2()