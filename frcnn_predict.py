"""
Calculates class specific and over-all evaluation scores on the model
Scores: TP,FN,FP,precision,recall,f2score
"""


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import random
import pprint
import sys
import time
import numpy as np
import pickle
import math
import cv2
import copy
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import pandas as pd
import os
import six

from sklearn.metrics import average_precision_score

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.objectives import categorical_crossentropy

from keras.models import Model
from keras.utils import generic_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
import argparse
import datetime
from definitions import *

#TODO: bbox_threshold needs to be defined

"""Predict Classes"""

def predict(C,model_rpn, model_classifier_only, class_mapping,test_base_path,bbox_threshold):

    classes = pd.DataFrame(columns=['image','pred_classes'])

    img_names = os.listdir(test_base_path)

    for img_name in img_names:
        #print('Get classes of {}/{}'.format(idx, len(test_imgs)))
        #img_name = img_path.split('/')[-1]
        #print(image_data['bboxes'])

        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        #print(img_path)
        st = time.time()

        '''Predict'''
        img_path = os.path.join(test_base_path, img_name)
        img = cv2.imread(img_path)

        X, ratio = format_img(img,C)  # X: normiertes Bild (kurze Seite 300 pixel), ratio = 300/Originallänge kurze Seite

        X = np.transpose(X, (0, 2, 3, 1))

        # get output layer Y1, Y2 from the RPN and the feature maps F
        # Y1: y_rpn_cls
        # Y2: y_rpn_regr
        [Y1, Y2, F] = model_rpn.predict(X)

        # Get bboxes by applying NMS
        # R.shape = (300, 4)
        R = rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)  # num_rois Boxen auswählen
            if ROIs.shape[1] == 0:  # wennn ROIs leer, fertig
                break

            if jk == R.shape[0] // C.num_rois:  # wenn ROIs nicht ganz aufgefüllt, auffüllen
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            # Calculate bboxes coordinates on resized image
            for ii in range(P_cls.shape[1]):

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                # Ignore 'bg' class
                if np.max(P_cls[0, ii, :]) < bbox_threshold[cls_name] or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append(
                    [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))


        pred_class_list = []  #contains all predicted class instances

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)

            for jk in range(new_boxes.shape[0]):
                pred_class_list.append(key)


        new_row = {'image': img_name, 'pred_classes': pred_class_list}
        classes = classes.append(new_row, ignore_index=True)

    return classes




'''---------------------- main----------------------------'''

if __name__ == '__main__':
    """General settings"""
    parser = argparse.ArgumentParser(description='The following parameters can be assigned:')
    parser.add_argument('--session_name', required=True, type=str)
    parser.add_argument('--base_path', required=True, type=str)
    parser.add_argument('--test_base_path', required=True, type=str)
    parser.add_argument('--out_path', required=True, type=str)
    parser.add_argument('--threshold_path', required=False,default=None, type=str)
    args = parser.parse_args()

    base_path = args.base_path  # path config and models are stored in
    test_base_path = args.test_base_path  # directory containing the pictures that are to predict
    threshold_path = args.threshold_path  # path to the thresholds (minimum probability for a class to be output)
    output_path = os.path.join(base_path, 'sessions', args.session_name)
    predict_store_path = os.path.join(args.out_path, "Prediction on {}".format(
        datetime.datetime.now().strftime("%A, %d %b %Y,%H %M")))  # path to save output figures in
    classes_path = os.path.join(predict_store_path, 'predicted_classes.csv')


    print('This is a Prediction Session of ->{}<-.'.format(args.session_name))
    print('Base Path: {}'.format(base_path))
    print('Output: {}'.format(predict_store_path))



    '''Prepare Model'''
    """Define Config"""
    config_output_filename = os.path.join(output_path, 'model', 'model_vgg_config.pickle')
    assert (os.path.exists(
        config_output_filename)), "Config File {} missing, Check if training has been performed with given session name".format(
        config_output_filename)
    os.makedirs(predict_store_path)

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    #Load thresholds
    threshold_df = pd.read_csv(threshold_path)
    print(threshold_df)
    threshold=threshold_df.to_dict('index')[0]

    print('Using Thresholds of file {}'.format(threshold_path))
    print('Thresholds{}'.format(threshold))


    # Load the records
    record_df = pd.read_csv(C.record_path)

    r_epochs = len(record_df)

    num_features = 512

    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers = nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = rpn_layer(shared_layers, num_anchors)

    classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    # Switch key value for class mapping
    class_mapping = C.class_mapping
    class_mapping = {v: k for k, v in class_mapping.items()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}


    '''------------Predict-----------------'''

    classes = predict(C,model_rpn, model_classifier_only,class_mapping, test_base_path, threshold)
    classes.to_csv(classes_path,sep=';', index=0)


