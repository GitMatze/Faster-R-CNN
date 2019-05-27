'''
Script for calculating the best bbox_tresholds (minimum output prob to count an object as found) for each class
'''

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



def optimize_bbox_thresh(possible_threshs, test_imgs):
    #param possibile_treshs: all the bbox thresholds that we want to compute the f2-score of
    score_dict = {}    # will store the f2-scores for every threshhold for every class
    score_dict['ALL']={'Class name':'ALL'}
    threshold_dict={} # will store the threshold with maximum f2-score for every class
    for cls_name in C.class_mapping:
        score_dict[cls_name]={'Class name':cls_name}  # convenient for later visualization

        for thresh in possible_threshs:
            score_dict[cls_name][thresh] = float('nan')


    for thresh in possible_threshs:
        print('Analyzing Threshold : {}'.format(thresh))
        classes = predict_classes(C,model_rpn, model_classifier_only,class_mapping,test_imgs, thresh)
        _, _, _, _, _, f2_score = get_scores(C,classes)

        for cls_name in C.class_mapping:
            score_dict[cls_name][thresh]= f2_score[cls_name]
        score_dict['ALL'][thresh] = f2_score['ALL']

        #save provisional result
        f2_scores_df = pd.DataFrame(data=score_dict).T
        ax = render_pandas_dataframe(f2_scores_df, header_columns=0, col_width=3.0)
        plt.suptitle('Class-specific F2 Scores, Zwischenergebnis \nSession: {}'.format(args.session_name),
                     fontsize=20)
        plt.savefig(os.path.join(store_path, 'optim_scores.pdf'))

    # now get the best f2-score for each class and save the respective bbox treshold in threshold_dict
    for cls_name in C.class_mapping:
        v = list(score_dict[cls_name].values())
        k = list(score_dict[cls_name].keys())
        threshold_dict[cls_name] = k[v.index(max(v[1:]))]  # ignore the class_name item

    return score_dict, threshold_dict



if __name__ == '__main__':
    """General settings"""
    parser = argparse.ArgumentParser(description='The following parameters can be assigned:')
    parser.add_argument('--session_name', required=True, type=str)
    parser.add_argument('--base_path', required=True, type=str)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--anno_path_optim', required=True, type=str)
    parser.add_argument('--anno_path_test', required=True, type=str)
    args = parser.parse_args()

    base_path = args.base_path  # path config and models are stored in
    data_path = args.data_path  # path test and train directory are stored in
    anno_path_test = args.anno_path_test  # path the test file is stored in
    anno_path_optim = args.anno_path_optim  #path the anno file for optimation is stored in
    output_path = os.path.join(base_path, 'sessions', args.session_name)
    store_path = os.path.join(output_path, "Optimization on {}".format(
        datetime.datetime.now().strftime("%A, %d %b %Y,%H %M")))  # path to save output figures in
    f2_scores_path = os.path.join(store_path, 'f2_scores.csv')


    print('This is an Optimization Session of ->{}<-.'.format(args.session_name))
    print('Base Path: {}'.format(base_path))
    print('Annotation File for Testing: {}'.format(anno_path_test))
    print('Image Data: {}'.format(data_path))
    print('Output: {}'.format(store_path))



    '''Prepare Model'''
    """Define Config"""
    config_output_filename = os.path.join(output_path, 'model', 'model_vgg_config.pickle')
    assert (os.path.exists(
        config_output_filename)), "Config File {} missing, Check if training has been performed with given session name".format(
        config_output_filename)
    os.makedirs(store_path)

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

    possible_threshs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    #possible_threshs = [0.1,0.9]
    optim_imgs, _, _ = get_data(anno_path_optim, data_path)
    score_dict, threshold_dict = optimize_bbox_thresh(possible_threshs, optim_imgs)
    print(threshold_dict)


    f2_scores_df = pd.DataFrame(data=score_dict).T
    threshold_df = pd.DataFrame(data=threshold_dict, index=[0])

    ax = render_pandas_dataframe(f2_scores_df, header_columns=0, col_width=3.0)

    plt.suptitle('Class-specific F2 Scores for different bbox_thresholds \nSession: {}'.format(args.session_name), fontsize=20)
    plt.savefig(os.path.join(store_path, 'optim_scores.pdf'))

    f2_scores_df.to_csv(f2_scores_path)
    threshold_df.to_csv(os.path.join(store_path, 'best_thresholds.csv'))



    #--------------------now a normal evaluation with optimized class thresholds
    test_imgs, _, _ = get_data(anno_path_test, data_path)
    classes = predict_classes(C, model_rpn, model_classifier_only, class_mapping, test_imgs, threshold_dict)
    TP, FP, FN, precision, recall, f2_score = get_scores(C, classes)

    scores = pd.DataFrame(columns=['Class', 'TP', 'FP', 'FN', 'F2-Score', 'Precision', 'Recall'])

    for key in precision:
        new_row = {'Class': key, 'TP': TP[key], 'FP': FP[key], 'FN': FN[key], 'F2-Score': f2_score[key],
                   'Precision': precision[key], 'Recall': recall[key]}
        scores = scores.append(new_row, ignore_index=True)

    ax = render_pandas_dataframe(scores, header_columns=0, col_width=3.0)
    plt.suptitle('Evaluation Scores of {}'.format(args.session_name), fontsize=30)

    plt.savefig(os.path.join(store_path, 'eval_scores.pdf'))
    #scores.to_csv(f2_score_path)
    #classes.to_csv(classes_path, index=0)

