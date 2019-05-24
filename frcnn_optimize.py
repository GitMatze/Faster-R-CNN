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
from frcnn_evaluate import *



def optimize_bbox_thresh():
    #initialize dictionary which will store score results
    possible_treshs = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    score_dict = {}
    for key in C.class_mapping:
        score_dict[key]={}
        for thresh in possible_treshs:
            score_dict[key][thresh] = 0






if __name__ == '__main__':
    """General settings"""
    parser = argparse.ArgumentParser(description='The following parameters can be assigned:')
    parser.add_argument('--session_name', required=True, type=str)
    parser.add_argument('--base_path', required=True, type=str)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--anno_path_test', required=True, type=str)
    args = parser.parse_args()

    base_path = args.base_path  # path config and models are stored in
    data_path = args.data_path  # path test and train directory are stored in
    anno_path_test = args.anno_path_test  # path the anno file is stored in
    output_path = os.path.join(base_path, 'sessions', args.session_name)
    test_store_path = os.path.join(output_path, "Evaluation Scores on {}".format(
        datetime.datetime.now().strftime("%A, %d %b %Y,%H %M")))  # path to save output figures in
    classes_path = os.path.join(test_store_path, 'predicted_classes.csv')
    f2_score_path = os.path.join(test_store_path, 'scores.csv')

    print('This is an Evaluation Session of ->{}<-.'.format(args.session_name))
    print('Base Path: {}'.format(base_path))
    print('Annotation File for Testing: {}'.format(anno_path_test))
    print('Image Data: {}'.format(data_path))
    print('Output: {}'.format(test_store_path))


    '''Prepare Model'''
    """Define Config"""
    config_output_filename = os.path.join(output_path, 'model', 'model_vgg_config.pickle')
    assert (os.path.exists(
        config_output_filename)), "Config File {} missing, Check if training has been performed with given session name".format(
        config_output_filename)
    os.makedirs(test_store_path)

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

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

    opt_scores = optimize_bbox_thresh()
