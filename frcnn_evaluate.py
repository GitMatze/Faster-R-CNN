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


"""Predict Classes"""

def predict_classes(C,model_rpn, model_classifier_only, class_mapping,test_imgs,bbox_threshold=0.5):

    # if single number, make dictionary with same threshold for every class
    if type(bbox_threshold)==float or type(bbox_threshold)== int:
        tresh = bbox_threshold
        bbox_threshold = {}
        for key in C.class_mapping:
            bbox_threshold[key] = tresh

    classes = pd.DataFrame(columns=['image','pred_classes', 'gt_classes'])

    for idx, image_data in enumerate(test_imgs):
        #print('Get classes of {}/{}'.format(idx, len(test_imgs)))
        img_path = image_data['filepath']
        img_name = img_path.split('/')[-1]
        #print(image_data['bboxes'])

        if not img_path.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        #print(img_path)
        st = time.time()

        '''Predict'''
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
        gt_class_list = []    #contains the ground truth class instances

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)

            for jk in range(new_boxes.shape[0]):
                pred_class_list.append(key)

        for gt_box in image_data['bboxes']:

            gt_class_list.append( gt_box['class'] )

        new_row = {'image': img_name, 'pred_classes': pred_class_list, 'gt_classes':gt_class_list}
        classes = classes.append(new_row, ignore_index=True)

    return classes



def get_scores(C,classes):
    TP = {}
    FN = {}
    FP = {}
    precision = {}
    recall ={}
    f2_score = {}
    #initialize classwise counters, key is a classname like 'Coat'
    for key in C.class_mapping:
        TP[key] = 0
        FN[key] = 0
        FP[key] = 0

    #ground_truth_copied = ground_truth.copy()
    for _,img in classes.iterrows():
        predictions = img['pred_classes']
        ground_truth = img['gt_classes'].copy() #removal of items needed, hence the copy

        #if prediction is found in ground truth, it's a true positive, otherwise a false positive
        for pred in predictions:
            if pred in ground_truth:
                TP[pred]+=1
                ground_truth.remove(pred)
            else:
                FP[pred]+=1

        for gt in ground_truth:
            FN[gt] += 1     #any remaining instance in ground truth was not found i.e. is a false negative


    for key in C.class_mapping:
        precision[key] = round(   TP[key] / (TP[key]+FP[key]+0.01)    ,2) ##a little hack to prevent division by zero errors
        recall[key]    = round(   TP[key] / (TP[key] + FN[key]+0.01)  ,2)
        f2_score[key]  = round(   5 * (precision[key] * recall[key]) / (4 * precision[key] + recall[key] + 0.01)  ,2)

    TP['ALL'] = sum(TP.values())
    FN['ALL'] = sum(FN.values())
    FP['ALL'] = sum(FP.values())
    precision['ALL'] = round(   TP['ALL'] / (TP['ALL'] + FP['ALL'] + 0.01) ,2)
    recall['ALL']    = round(   TP['ALL'] / (TP['ALL'] + FN['ALL']+0.01)   ,2)
    f2_score['ALL']  = round(  5 * (precision['ALL'] * recall['ALL']) / (4 * precision['ALL'] + recall['ALL'] + 0.01) ,2)


    print('Precision: {}, Recall: {}'.format(precision['ALL'], recall['ALL']))
    print('F2-Score: {}'.format(f2_score['ALL']))

    return TP,FP,FN,precision,recall,f2_score

'''renders a pandas dataframe'''
def render_pandas_dataframe(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax



'''---------------------- main----------------------------'''

if __name__ == '__main__':
    """General settings"""
    parser = argparse.ArgumentParser(description='The following parameters can be assigned:')
    parser.add_argument('--session_name', required=True, type=str)
    parser.add_argument('--base_path', required=True, type=str)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument('--anno_path_test', required=True, type=str)
    parser.add_argument('--threshold_path', required=False,default=None, type=str)
    parser.add_argument('--threshold', required=False, default=0.5, type=int)
    parser.add_argument('--keyword', required=False, default=None, type=str)   #is used for giving a hint on what was evalated on the output pdf
    args = parser.parse_args()

    base_path = args.base_path  # path config and models are stored in
    data_path = args.data_path  # path test and train directory are stored in
    anno_path_test = args.anno_path_test  # path the anno file is stored in
    threshold_path = args.threshold_path  # path to the thresholds (minimum probability for a class to be output)
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
    print('Keyword: {}'.format(args.keyword))


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

    #Load thresholds
    if threshold_path is not None:
        threshold_df = pd.read_csv(threshold_path)
        print(threshold_df)
        threshold=threshold_df.to_dict('index')[0]

        print('Using Thresholds of file {}'.format(threshold_path))
        print('Thresholds{}'.format(threshold))
    else:
        threshold =  args.threshold
        print('No threshold path given, thus using scalar value: {}'.format(args.threshold))


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

    '''Evaluate'''
    test_imgs, _, _ = get_data(anno_path_test, data_path)
    classes = predict_classes(C,model_rpn, model_classifier_only,class_mapping, test_imgs, threshold)
    TP,FP,FN,precision,recall,f2_score = get_scores(C,classes)

    scores = pd.DataFrame(columns=['Class', 'TP', 'FP', 'FN', 'F2-Score', 'Precision', 'Recall'])

    for key in precision:
        new_row = {'Class': key, 'TP': TP[key], 'FP': FP[key],'FN': FN[key],'F2-Score': f2_score[key],'Precision': precision[key], 'Recall':recall[key]}
        scores = scores.append(new_row, ignore_index=True)


    ax = render_pandas_dataframe(scores, header_columns=0, col_width=3.0)
    plt.suptitle('Evaluation Scores of {} \n(Keyword: {})'.format(args.session_name, args.keyword), fontsize = 30)

    plt.savefig(os.path.join(test_store_path, 'scores.pdf'))
    scores.to_csv(f2_score_path)
    classes.to_csv(classes_path, index=0)


