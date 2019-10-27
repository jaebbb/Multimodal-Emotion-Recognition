# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import cv2
import pickle
import time
import pylab as pl
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.special import expit

MODULE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
MODEL_DIR  = os.path.join(MODULE_DIR, "model")
MODEL_FILE = os.path.join(MODEL_DIR, "hr_res101.pkl")

DATA_DIR        = os.path.join(MODULE_DIR, "data")
DATA_INPUT_DIR  = os.path.join(DATA_DIR, "input")
DATA_OUTPUT_DIR = os.path.join(DATA_DIR, "output")

MAX_INPUT_DIM = 5000.0

class TinyFaceDetection(object):
    """
    Tiny Face Detector in TensorFlow + Face Selection
    https://github.com/cydonia999/Tiny_Faces_in_Tensorflow
    Hu Peiyun, Ramanan Deva, "Finding Tiny Faces," CVPR, 2017
    """
    def __init__(self, **kwargs):
        """
        Initialize TinyFace Detection
        """
        self.params = dict(weight_file_path=MODEL_FILE, gpu_memory_fraction=0.3, verbose=0, prob_thresh=0.5, nms_thresh=0.1, MAX_INPUT_DIM = 5000.0)
        self.params.update(kwargs)

        self.tinynet = tinyface_create_net(weight_file_path = self.params["weight_file_path"], gpu_memory_fraction = self.params["gpu_memory_fraction"])
        
        self.print_log("TinyFace", "Init Detector")
    # __init__


    # Singleton
    internal = None
    @classmethod
    def getDetector(cls, new_allocate = False, **kwargs):
        if cls.internal is None or new_allocate == True:
            cls.internal = cls(**kwargs)
        return cls.internal
    # getDetector

    def detect(self, image):
        """
        Input : Image (RGB, uint8)
        Output: bboxes[0:4]: bboxes, bboxes[4]: probabilities
        """
        start  = time.time()
        bboxes = tinyface_detect_face(self.tinynet, image, prob_thresh=self.params["prob_thresh"], nms_thresh=self.params["nms_thresh"], MAX_INPUT_DIM = self.params["MAX_INPUT_DIM"], verbose = self.params["verbose"])

        return bboxes, time.time() - start
    # detect

    def draw_bboxes(self, image, bboxes, lw = 3):
        overlay_bounding_boxes(image, bboxes, lw)
        pass
    # draw_bboxes

    def print_log(self, type, messege):
        if self.params["verbose"] == 1:
            print("[%s] %s"%(type, messege))
    # print_log
# TinyFace

def tinyface_draw_bboxes(image, bboxes, lw = 3, verbose = 1):
    image_result = image.copy()
    overlay_bounding_boxes(image_result, bboxes, lw)    
    if verbose == 1:
        plt.imshow(image_result)
        plt.show()
    # if
    return image_result
# tinyface_draw_bboxes

def tinyface_detect_face(tinynet, image, prob_thresh=0.5, nms_thresh=0.1, MAX_INPUT_DIM = 5000.0, verbose = 1):
    """
    Input:
    + image: RGB
    """
    image_f = image.astype(np.float32)

    def _calc_scales():
        raw_h, raw_w = image.shape[0], image.shape[1]
        min_scale = min(np.floor(np.log2(np.max(tinynet["clusters_w"][tinynet["normal_idx"]] / raw_w))),
                        np.floor(np.log2(np.max(tinynet["clusters_h"][tinynet["normal_idx"]] / raw_h))))
        max_scale = min(1.0, -np.log2(max(raw_h, raw_w) / MAX_INPUT_DIM))
        scales_down = np.arange(min_scale, 0, 1.)
        scales_up = np.arange(0.5, max_scale, 0.5)
        scales_pow = np.hstack((scales_down, scales_up))
        scales = np.power(2.0, scales_pow)
        return scales
    # _calc_scales

    scales = _calc_scales()
    if verbose == 1: start = time.time()

    # initialize output
    bboxes = np.empty(shape=(0, 5))

    # process input at different scales
    for s in scales:
        if verbose == 1: print("Processing at scale {:.4f}".format(s))

        img = cv2.resize(image_f, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        img = img - tinynet["average_image"]
        img = img[np.newaxis, :]

        # we don't run every template on every scale ids of templates to ignore
        tids = list(range(4, 12)) + ([] if s <= 1.0 else list(range(18, 25)))
        ignoredTids = list(set(range(0, tinynet["clusters"].shape[0])) - set(tids))

        # run through the net
        score_final_tf = tinynet["tinynet_fun"](img)

        # collect scores
        score_cls_tf, score_reg_tf = score_final_tf[:, :, :, :25], score_final_tf[:, :, :, 25:125]
        prob_cls_tf = expit(score_cls_tf)
        prob_cls_tf[0, :, :, ignoredTids] = 0.0

        def _calc_bounding_boxes():
            # threshold for detection
            _, fy, fx, fc = np.where(prob_cls_tf > prob_thresh)

            # interpret heatmap into bounding boxes
            cy = fy * 8 - 1
            cx = fx * 8 - 1
            ch = tinynet["clusters"][fc, 3] - tinynet["clusters"][fc, 1] + 1
            cw = tinynet["clusters"][fc, 2] - tinynet["clusters"][fc, 0] + 1

            # extract bounding box refinement
            Nt = tinynet["clusters"].shape[0]
            tx = score_reg_tf[0, :, :, 0:Nt]
            ty = score_reg_tf[0, :, :, Nt:2*Nt]
            tw = score_reg_tf[0, :, :, 2*Nt:3*Nt]
            th = score_reg_tf[0, :, :, 3*Nt:4*Nt]

            # refine bounding boxes
            dcx = cw * tx[fy, fx, fc]
            dcy = ch * ty[fy, fx, fc]
            rcx = cx + dcx
            rcy = cy + dcy
            rcw = cw * np.exp(tw[fy, fx, fc])
            rch = ch * np.exp(th[fy, fx, fc])

            scores = score_cls_tf[0, fy, fx, fc]
            tmp_bboxes = np.vstack((rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2))
            tmp_bboxes = np.vstack((tmp_bboxes / s, scores))
            tmp_bboxes = tmp_bboxes.transpose()
            return tmp_bboxes
        # _calc_bounding_boxes

        tmp_bboxes = _calc_bounding_boxes()
        bboxes = np.vstack((bboxes, tmp_bboxes)) # <class 'tuple'>: (5265, 5)
    # for

    if verbose == 1: print("Time {:.2f} secs".format(time.time() - start))

    # non maximum suppression
    refind_idx = tinynet["nms_thresh_fun"](bboxes, nms_thresh)
    refined_bboxes = bboxes[refind_idx]

    refined_bboxes = np.hstack([refined_bboxes[:, 0:2], refined_bboxes[:, 2:4] - refined_bboxes[:, 0:2] + 1, refined_bboxes[:, 4:]])

    return refined_bboxes
# tinyface_detect_face

def tinyface_create_net(weight_file_path = MODEL_FILE, gpu_memory_fraction = 0.3):
    """
    Do Nhu Tai
    Load tinyface model and return tinyface model
    """
    with tf.Graph().as_default():
        with tf.variable_scope('tinyface', reuse = tf.AUTO_REUSE):     
            # session
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

            # placeholder of input images. Currently batch size of one is supported.
            tf_input = tf.placeholder(tf.float32, [1, None, None, 3], name = "input") # n, h, w, c

            # Create the tiny face model which weights are loaded from a pretrained model.
            model = Model(weight_file_path)
            tf_score_final = model.tiny_face(tf_input)

            # non maximum suppression
            tf_bboxes      = tf.placeholder(tf.float32, [None, 5],    name = "bboxes") # [num_boxes, 4]
            tf_nms_thresh  = tf.placeholder(tf.float32, shape = None, name = "nms_thresh") # [num_boxes, 4]
            tf_refind_idx  = tf.image.non_max_suppression(tf_bboxes[:, :4], tf_bboxes[:, 4], max_output_size = tf.shape(tf_bboxes)[0], iou_threshold = tf_nms_thresh, name = "refind_idx")

            nms_thresh_fun = lambda bboxes, nms_thresh : sess.run(tf_refind_idx, feed_dict={tf_bboxes:bboxes, tf_nms_thresh: nms_thresh})

            tinynet_fun = lambda img : sess.run(tf_score_final, feed_dict={tf_input:img})

            sess.run(tf.global_variables_initializer())
        # with
    # with

    # Load an average image and clusters(reference boxes of templates).
    average_image = model.get_data_by_key("average_image")
    clusters = model.get_data_by_key("clusters")
    clusters_h = clusters[:, 3] - clusters[:, 1] + 1
    clusters_w = clusters[:, 2] - clusters[:, 0] + 1
    normal_idx = np.where(clusters[:, 4] == 1)

    return {"sess": sess, "model": model, "tinynet_fun": tinynet_fun, "tf_input": tf_input, "tinynet_fun": tinynet_fun, "nms_thresh_fun": nms_thresh_fun,
            "average_image": average_image, "clusters_h": clusters_h, "clusters_w": clusters_w, "normal_idx": normal_idx, "clusters": clusters}
# tinyface_create_net

def evaluate(weight_file_path = MODEL_FILE, data_dir = DATA_INPUT_DIR, output_dir = DATA_OUTPUT_DIR, prob_thresh=0.5, nms_thresh=0.1, lw=3, MAX_INPUT_DIM = 5000.0, display=False):
    """Detect faces in images.
    Args:
      prob_thresh:
          The threshold of detection confidence.
      nms_thresh:
          The overlap threshold of non maximum suppression
      weight_file_path:
          A pretrained weight file in the pickle format
          generated by matconvnet_hr101_to_tf.py.
      data_dir:
          A directory which contains images.
      output_dir:
          A directory into which images with detected faces are output.
      lw:
          Line width of bounding boxes. If zero specified,
          this is determined based on confidence of each detection.
      display:
          Display tiny face images on window.
    Returns:
      None.
    """
    if os.path.exists(output_dir) == False: os.makedirs(output_dir)
    
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        with tf.variable_scope('tinyface', reuse = tf.AUTO_REUSE):     
            # placeholder of input images. Currently batch size of one is supported.
            x = tf.placeholder(tf.float32, [1, None, None, 3], name = "input") # n, h, w, c

            # Create the tiny face model which weights are loaded from a pretrained model.
            model = Model(weight_file_path)
            score_final = model.tiny_face(x)
            score_final_fun = lambda img : sess.run(score_final, feed_dict={x:img})
            
            

    # Find image files in data_dir.
    filenames = []
    for ext in ('*.png', '*.gif', '*.jpg', '*.jpeg'):
        filenames.extend(glob.glob(os.path.join(data_dir, ext)))

    # with open(weight_file_path, "rb") as f:
    #    _, mat_params_dict = pickle.load(f)

    # Load an average image and clusters(reference boxes of templates).
    average_image = model.get_data_by_key("average_image")
    clusters = model.get_data_by_key("clusters")
    clusters_h = clusters[:, 3] - clusters[:, 1] + 1
    clusters_w = clusters[:, 2] - clusters[:, 0] + 1
    normal_idx = np.where(clusters[:, 4] == 1)

    # main
    with tf.Session(graph = tf_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for filename in filenames:
            fname = filename.split(os.sep)[-1]
            raw_img = cv2.imread(filename)
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            raw_img_f = raw_img.astype(np.float32)

            def _calc_scales():
                raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]
                min_scale = min(np.floor(np.log2(np.max(clusters_w[normal_idx] / raw_w))),
                                np.floor(np.log2(np.max(clusters_h[normal_idx] / raw_h))))
                max_scale = min(1.0, -np.log2(max(raw_h, raw_w) / MAX_INPUT_DIM))
                scales_down = np.arange(min_scale, 0, 1.)
                scales_up = np.arange(0.5, max_scale, 0.5)
                scales_pow = np.hstack((scales_down, scales_up))
                scales = np.power(2.0, scales_pow)
                return scales

            scales = _calc_scales()
            start = time.time()

            # initialize output
            bboxes = np.empty(shape=(0, 5))

            # process input at different scales
            for s in scales:
                print("Processing {} at scale {:.4f}".format(fname, s))
                img = cv2.resize(raw_img_f, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                img = img - average_image
                img = img[np.newaxis, :]

                # we don't run every template on every scale ids of templates to ignore
                tids = list(range(4, 12)) + ([] if s <= 1.0 else list(range(18, 25)))
                ignoredTids = list(set(range(0, clusters.shape[0])) - set(tids))

                # run through the net
                # score_final_tf = sess.run(score_final, feed_dict={x: img})
                score_final_tf = score_final_fun(img)

                # collect scores
                score_cls_tf, score_reg_tf = score_final_tf[:, :, :, :25], score_final_tf[:, :, :, 25:125]
                prob_cls_tf = expit(score_cls_tf)
                prob_cls_tf[0, :, :, ignoredTids] = 0.0

                def _calc_bounding_boxes():
                    # threshold for detection
                    _, fy, fx, fc = np.where(prob_cls_tf > prob_thresh)

                    # interpret heatmap into bounding boxes
                    cy = fy * 8 - 1
                    cx = fx * 8 - 1
                    ch = clusters[fc, 3] - clusters[fc, 1] + 1
                    cw = clusters[fc, 2] - clusters[fc, 0] + 1

                    # extract bounding box refinement
                    Nt = clusters.shape[0]
                    tx = score_reg_tf[0, :, :, 0:Nt]
                    ty = score_reg_tf[0, :, :, Nt:2*Nt]
                    tw = score_reg_tf[0, :, :, 2*Nt:3*Nt]
                    th = score_reg_tf[0, :, :, 3*Nt:4*Nt]

                    # refine bounding boxes
                    dcx = cw * tx[fy, fx, fc]
                    dcy = ch * ty[fy, fx, fc]
                    rcx = cx + dcx
                    rcy = cy + dcy
                    rcw = cw * np.exp(tw[fy, fx, fc])
                    rch = ch * np.exp(th[fy, fx, fc])

                    scores = score_cls_tf[0, fy, fx, fc]
                    tmp_bboxes = np.vstack((rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2))
                    tmp_bboxes = np.vstack((tmp_bboxes / s, scores))
                    tmp_bboxes = tmp_bboxes.transpose()
                    return tmp_bboxes

                tmp_bboxes = _calc_bounding_boxes()
                bboxes = np.vstack((bboxes, tmp_bboxes)) # <class 'tuple'>: (5265, 5)
            # for


            print("time {:.2f} secs for {}".format(time.time() - start, fname))

            # non maximum suppression
            # refind_idx = util.nms(bboxes, nms_thresh)
            refind_idx = tf.image.non_max_suppression(tf.convert_to_tensor(bboxes[:, :4], dtype=tf.float32),
                                                         tf.convert_to_tensor(bboxes[:, 4], dtype=tf.float32),
                                                         max_output_size=bboxes.shape[0], iou_threshold=nms_thresh)
            refind_idx = sess.run(refind_idx)
            refined_bboxes = bboxes[refind_idx]
            refined_bboxes = np.hstack([refined_bboxes[:, 0:2], refined_bboxes[:, 2:4] - refined_bboxes[:, 0:2] + 1, refined_bboxes[:, 4:]])

            overlay_bounding_boxes(raw_img, refined_bboxes, lw)

            if display:
                # plt.axis('off')
                plt.imshow(raw_img)
                plt.show()
            # if

            # save image with bounding boxes
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, fname), raw_img)
        # for
    # with
# evaluate		

def overlay_bounding_boxes(raw_img, refined_bboxes, lw):
    """Overlay bounding boxes of face on images.
      Args:
        raw_img:
          A target image.
        refined_bboxes:
          Bounding boxes of detected faces.
        lw:
          Line width of bounding boxes. If zero specified,
          this is determined based on confidence of each detection.
      Returns:
        None.
    """

    # Overlay bounding boxes on an image with the color based on the confidence.
    for r in refined_bboxes:
        _score = expit(r[4])
        cm_idx = int(np.ceil(_score * 255))
        rect_color = [int(np.ceil(x * 255)) for x in cm_data[cm_idx]]  # parula
        _lw = lw
        if lw == 0:  # line width of each bounding box is adaptively determined.
            # bw, bh = r[2] - r[0] + 1, r[3] - r[0] + 1
            bw, bh = r[2], r[3]
            _lw = 1 if min(bw, bh) <= 20 else max(2, min(3, min(bh / 20, bw / 20)))
            _lw = int(np.ceil(_lw * _score))

        _r = [int(x) for x in r[:4]]
        cv2.rectangle(raw_img, (_r[0], _r[1]), (_r[0] + _r[2], _r[1] + _r[3]), rect_color, _lw)
# overlay_bounding_boxes	

def nms(dets, prob_thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= prob_thresh)[0]

        order = order[inds + 1]
    return keep
# nms

# colormap parula borrowed from
# https://github.com/BIDS/colormap/blob/master/fake_parula.py
cm_data = [[ 0.26710521,  0.03311059,  0.6188155 ],
       [ 0.26493929,  0.04780926,  0.62261795],
       [ 0.26260545,  0.06084214,  0.62619176],
       [ 0.26009691,  0.07264411,  0.62951561],
       [ 0.25740785,  0.08360391,  0.63256745],
       [ 0.25453369,  0.09395358,  0.63532497],
       [ 0.25147146,  0.10384228,  0.6377661 ],
       [ 0.24822014,  0.11337029,  0.6398697 ],
       [ 0.24478105,  0.12260661,  0.64161629],
       [ 0.24115816,  0.131599  ,  0.6429888 ],
       [ 0.23735836,  0.14038009,  0.64397346],
       [ 0.23339166,  0.14897137,  0.64456048],
       [ 0.22927127,  0.15738602,  0.64474476],
       [ 0.22501278,  0.16563165,  0.64452595],
       [ 0.22063349,  0.17371215,  0.64390834],
       [ 0.21616055,  0.18162302,  0.64290515],
       [ 0.21161851,  0.18936156,  0.64153295],
       [ 0.20703353,  0.19692415,  0.63981287],
       [ 0.20243273,  0.20430706,  0.63776986],
       [ 0.19784363,  0.211507  ,  0.63543183],
       [ 0.19329361,  0.21852157,  0.63282872],
       [ 0.18880937,  0.2253495 ,  0.62999156],
       [ 0.18442119,  0.23198815,  0.62695569],
       [ 0.18014936,  0.23844124,  0.62374886],
       [ 0.17601569,  0.24471172,  0.62040016],
       [ 0.17204028,  0.25080356,  0.61693715],
       [ 0.16824123,  0.25672163,  0.6133854 ],
       [ 0.16463462,  0.26247158,  0.60976836],
       [ 0.16123449,  0.26805963,  0.60610723],
       [ 0.15805279,  0.27349243,  0.60242099],
       [ 0.15509948,  0.27877688,  0.59872645],
       [ 0.15238249,  0.28392004,  0.59503836],
       [ 0.14990781,  0.28892902,  0.59136956],
       [ 0.14767951,  0.29381086,  0.58773113],
       [ 0.14569979,  0.29857245,  0.58413255],
       [ 0.1439691 ,  0.30322055,  0.58058191],
       [ 0.14248613,  0.30776167,  0.57708599],
       [ 0.14124797,  0.31220208,  0.57365049],
       [ 0.14025018,  0.31654779,  0.57028011],
       [ 0.13948691,  0.32080454,  0.5669787 ],
       [ 0.13895174,  0.32497744,  0.56375063],
       [ 0.13863958,  0.32907012,  0.56060453],
       [ 0.138537  ,  0.3330895 ,  0.55753513],
       [ 0.13863384,  0.33704026,  0.55454374],
       [ 0.13891931,  0.34092684,  0.55163126],
       [ 0.13938212,  0.34475344,  0.54879827],
       [ 0.14001061,  0.34852402,  0.54604503],
       [ 0.14079292,  0.35224233,  0.54337156],
       [ 0.14172091,  0.35590982,  0.54078769],
       [ 0.14277848,  0.35953205,  0.53828312],
       [ 0.14395358,  0.36311234,  0.53585661],
       [ 0.1452346 ,  0.36665374,  0.5335074 ],
       [ 0.14661019,  0.3701591 ,  0.5312346 ],
       [ 0.14807104,  0.37363011,  0.52904278],
       [ 0.1496059 ,  0.3770697 ,  0.52692951],
       [ 0.15120289,  0.3804813 ,  0.52488853],
       [ 0.15285214,  0.38386729,  0.52291854],
       [ 0.15454421,  0.38722991,  0.52101815],
       [ 0.15627225,  0.39056998,  0.5191937 ],
       [ 0.15802555,  0.39389087,  0.5174364 ],
       [ 0.15979549,  0.39719482,  0.51574311],
       [ 0.16157425,  0.40048375,  0.51411214],
       [ 0.16335571,  0.40375871,  0.51254622],
       [ 0.16513234,  0.40702178,  0.51104174],
       [ 0.1668964 ,  0.41027528,  0.50959299],
       [ 0.16864151,  0.41352084,  0.50819797],
       [ 0.17036277,  0.41675941,  0.50685814],
       [ 0.1720542 ,  0.41999269,  0.50557008],
       [ 0.17370932,  0.42322271,  0.50432818],
       [ 0.17532301,  0.42645082,  0.50313007],
       [ 0.17689176,  0.42967776,  0.50197686],
       [ 0.17841013,  0.43290523,  0.5008633 ],
       [ 0.17987314,  0.43613477,  0.49978492],
       [ 0.18127676,  0.43936752,  0.49873901],
       [ 0.18261885,  0.44260392,  0.49772638],
       [ 0.18389409,  0.44584578,  0.49673978],
       [ 0.18509911,  0.44909409,  0.49577605],
       [ 0.18623135,  0.4523496 ,  0.494833  ],
       [ 0.18728844,  0.45561305,  0.49390803],
       [ 0.18826671,  0.45888565,  0.49299567],
       [ 0.18916393,  0.46216809,  0.49209268],
       [ 0.18997879,  0.46546084,  0.49119678],
       [ 0.19070881,  0.46876472,  0.49030328],
       [ 0.19135221,  0.47208035,  0.48940827],
       [ 0.19190791,  0.47540815,  0.48850845],
       [ 0.19237491,  0.47874852,  0.4876002 ],
       [ 0.19275204,  0.48210192,  0.48667935],
       [ 0.19303899,  0.48546858,  0.48574251],
       [ 0.19323526,  0.48884877,  0.48478573],
       [ 0.19334062,  0.49224271,  0.48380506],
       [ 0.19335574,  0.49565037,  0.4827974 ],
       [ 0.19328143,  0.49907173,  0.48175948],
       [ 0.19311664,  0.50250719,  0.48068559],
       [ 0.192864  ,  0.50595628,  0.47957408],
       [ 0.19252521,  0.50941877,  0.47842186],
       [ 0.19210087,  0.51289469,  0.47722441],
       [ 0.19159194,  0.516384  ,  0.47597744],
       [ 0.19100267,  0.51988593,  0.47467988],
       [ 0.19033595,  0.52340005,  0.47332894],
       [ 0.18959113,  0.5269267 ,  0.47191795],
       [ 0.18877336,  0.530465  ,  0.47044603],
       [ 0.18788765,  0.53401416,  0.46891178],
       [ 0.18693822,  0.53757359,  0.46731272],
       [ 0.18592276,  0.54114404,  0.46563962],
       [ 0.18485204,  0.54472367,  0.46389595],
       [ 0.18373148,  0.5483118 ,  0.46207951],
       [ 0.18256585,  0.55190791,  0.4601871 ],
       [ 0.18135481,  0.55551253,  0.45821002],
       [ 0.18011172,  0.55912361,  0.45615277],
       [ 0.17884392,  0.56274038,  0.45401341],
       [ 0.17755858,  0.56636217,  0.45178933],
       [ 0.17625543,  0.56998972,  0.44946971],
       [ 0.174952  ,  0.57362064,  0.44706119],
       [ 0.17365805,  0.57725408,  0.44456198],
       [ 0.17238403,  0.58088916,  0.4419703 ],
       [ 0.17113321,  0.58452637,  0.43927576],
       [ 0.1699221 ,  0.58816399,  0.43648119],
       [ 0.1687662 ,  0.5918006 ,  0.43358772],
       [ 0.16767908,  0.59543526,  0.43059358],
       [ 0.16667511,  0.59906699,  0.42749697],
       [ 0.16575939,  0.60269653,  0.42428344],
       [ 0.16495764,  0.6063212 ,  0.42096245],
       [ 0.16428695,  0.60993988,  0.41753246],
       [ 0.16376481,  0.61355147,  0.41399151],
       [ 0.16340924,  0.61715487,  0.41033757],
       [ 0.16323549,  0.62074951,  0.40656329],
       [ 0.16326148,  0.62433443,  0.40266378],
       [ 0.16351136,  0.62790748,  0.39864431],
       [ 0.16400433,  0.63146734,  0.39450263],
       [ 0.16475937,  0.63501264,  0.39023638],
       [ 0.16579502,  0.63854196,  0.38584309],
       [ 0.16712921,  0.64205381,  0.38132023],
       [ 0.168779  ,  0.64554661,  0.37666513],
       [ 0.17075915,  0.64901912,  0.37186962],
       [ 0.17308572,  0.65246934,  0.36693299],
       [ 0.1757732 ,  0.65589512,  0.36185643],
       [ 0.17883344,  0.65929449,  0.3566372 ],
       [ 0.18227669,  0.66266536,  0.35127251],
       [ 0.18611159,  0.66600553,  0.34575959],
       [ 0.19034516,  0.66931265,  0.34009571],
       [ 0.19498285,  0.67258423,  0.3342782 ],
       [ 0.20002863,  0.67581761,  0.32830456],
       [ 0.20548509,  0.67900997,  0.3221725 ],
       [ 0.21135348,  0.68215834,  0.31587999],
       [ 0.2176339 ,  0.68525954,  0.30942543],
       [ 0.22432532,  0.68831023,  0.30280771],
       [ 0.23142568,  0.69130688,  0.29602636],
       [ 0.23893914,  0.69424565,  0.28906643],
       [ 0.2468574 ,  0.69712255,  0.28194103],
       [ 0.25517514,  0.69993351,  0.27465372],
       [ 0.26388625,  0.70267437,  0.26720869],
       [ 0.27298333,  0.70534087,  0.25961196],
       [ 0.28246016,  0.70792854,  0.25186761],
       [ 0.29232159,  0.71043184,  0.2439642 ],
       [ 0.30253943,  0.71284765,  0.23594089],
       [ 0.31309875,  0.71517209,  0.22781515],
       [ 0.32399522,  0.71740028,  0.21959115],
       [ 0.33520729,  0.71952906,  0.21129816],
       [ 0.3467003 ,  0.72155723,  0.20298257],
       [ 0.35846225,  0.72348143,  0.19466318],
       [ 0.3704552 ,  0.72530195,  0.18639333],
       [ 0.38264126,  0.72702007,  0.17822762],
       [ 0.39499483,  0.72863609,  0.17020921],
       [ 0.40746591,  0.73015499,  0.1624122 ],
       [ 0.42001969,  0.73158058,  0.15489659],
       [ 0.43261504,  0.73291878,  0.14773267],
       [ 0.44521378,  0.73417623,  0.14099043],
       [ 0.45777768,  0.73536072,  0.13474173],
       [ 0.47028295,  0.73647823,  0.1290455 ],
       [ 0.48268544,  0.73753985,  0.12397794],
       [ 0.49497773,  0.73854983,  0.11957878],
       [ 0.5071369 ,  0.73951621,  0.11589589],
       [ 0.51913764,  0.74044827,  0.11296861],
       [ 0.53098624,  0.74134823,  0.11080237],
       [ 0.5426701 ,  0.74222288,  0.10940411],
       [ 0.55417235,  0.74308049,  0.10876749],
       [ 0.56550904,  0.74392086,  0.10885609],
       [ 0.57667994,  0.74474781,  0.10963233],
       [ 0.58767906,  0.74556676,  0.11105089],
       [ 0.59850723,  0.74638125,  0.1130567 ],
       [ 0.609179  ,  0.74719067,  0.11558918],
       [ 0.61969877,  0.74799703,  0.11859042],
       [ 0.63007148,  0.74880206,  0.12200388],
       [ 0.64030249,  0.74960714,  0.12577596],
       [ 0.65038997,  0.75041586,  0.12985641],
       [ 0.66034774,  0.75122659,  0.1342004 ],
       [ 0.67018264,  0.75203968,  0.13876817],
       [ 0.67990043,  0.75285567,  0.14352456],
       [ 0.68950682,  0.75367492,  0.14843886],
       [ 0.69900745,  0.75449768,  0.15348445],
       [ 0.70840781,  0.75532408,  0.15863839],
       [ 0.71771325,  0.75615416,  0.16388098],
       [ 0.72692898,  0.75698787,  0.1691954 ],
       [ 0.73606001,  0.75782508,  0.17456729],
       [ 0.74511119,  0.75866562,  0.17998443],
       [ 0.75408719,  0.75950924,  0.18543644],
       [ 0.76299247,  0.76035568,  0.19091446],
       [ 0.77183123,  0.76120466,  0.19641095],
       [ 0.78060815,  0.76205561,  0.20191973],
       [ 0.78932717,  0.76290815,  0.20743538],
       [ 0.79799213,  0.76376186,  0.21295324],
       [ 0.8066067 ,  0.76461631,  0.21846931],
       [ 0.81517444,  0.76547101,  0.22398014],
       [ 0.82369877,  0.76632547,  0.2294827 ],
       [ 0.832183  ,  0.7671792 ,  0.2349743 ],
       [ 0.8406303 ,  0.76803167,  0.24045248],
       [ 0.84904371,  0.76888236,  0.24591492],
       [ 0.85742615,  0.76973076,  0.25135935],
       [ 0.86578037,  0.77057636,  0.25678342],
       [ 0.87410891,  0.77141875,  0.2621846 ],
       [ 0.88241406,  0.77225757,  0.26755999],
       [ 0.89070781,  0.77308772,  0.27291122],
       [ 0.89898836,  0.77391069,  0.27823228],
       [ 0.90725475,  0.77472764,  0.28351668],
       [ 0.91550775,  0.77553893,  0.28875751],
       [ 0.92375722,  0.7763404 ,  0.29395046],
       [ 0.9320227 ,  0.77712286,  0.29909267],
       [ 0.94027715,  0.7779011 ,  0.30415428],
       [ 0.94856742,  0.77865213,  0.3091325 ],
       [ 0.95686038,  0.7793949 ,  0.31397459],
       [ 0.965222  ,  0.7800975 ,  0.31864342],
       [ 0.97365189,  0.78076521,  0.32301107],
       [ 0.98227405,  0.78134549,  0.32678728],
       [ 0.99136564,  0.78176999,  0.3281624 ],
       [ 0.99505988,  0.78542889,  0.32106514],
       [ 0.99594185,  0.79046888,  0.31648808],
       [ 0.99646635,  0.79566972,  0.31244662],
       [ 0.99681528,  0.80094905,  0.30858532],
       [ 0.9970578 ,  0.80627441,  0.30479247],
       [ 0.99724883,  0.81161757,  0.30105328],
       [ 0.99736711,  0.81699344,  0.29725528],
       [ 0.99742254,  0.82239736,  0.29337235],
       [ 0.99744736,  0.82781159,  0.28943391],
       [ 0.99744951,  0.83323244,  0.28543062],
       [ 0.9973953 ,  0.83867931,  0.2812767 ],
       [ 0.99727248,  0.84415897,  0.27692897],
       [ 0.99713953,  0.84963903,  0.27248698],
       [ 0.99698641,  0.85512544,  0.26791703],
       [ 0.99673736,  0.86065927,  0.26304767],
       [ 0.99652358,  0.86616957,  0.25813608],
       [ 0.99622774,  0.87171946,  0.25292044],
       [ 0.99590494,  0.87727931,  0.24750009],
       [ 0.99555225,  0.88285068,  0.2418514 ],
       [ 0.99513763,  0.8884501 ,  0.23588062],
       [ 0.99471252,  0.89405076,  0.2296837 ],
       [ 0.99421873,  0.89968246,  0.2230963 ],
       [ 0.99370185,  0.90532165,  0.21619768],
       [ 0.99313786,  0.91098038,  0.2088926 ],
       [ 0.99250707,  0.91666811,  0.20108214],
       [ 0.99187888,  0.92235023,  0.19290417],
       [ 0.99110991,  0.92809686,  0.18387963],
       [ 0.99042108,  0.93379995,  0.17458127],
       [ 0.98958484,  0.93956962,  0.16420166],
       [ 0.98873988,  0.94533859,  0.15303117],
       [ 0.98784836,  0.95112482,  0.14074826],
       [ 0.98680727,  0.95697596,  0.12661626]]

class Model(object):
    def __init__(self, weight_file_path, device_string = "gpu:0"):
        """Overlay bounding boxes of face on images.
          Args:
            weight_file_path:
                A pretrained weight file in the pickle format
                generated by matconvnet_hr101_to_tf.py.
          Returns:
            None.
        """
        self.dtype = tf.float32
        self.weight_file_path = weight_file_path
        self.device_string = device_string
        with open(self.weight_file_path, "rb") as f:
            self.mat_blocks_dict, self.mat_params_dict = pickle.load(f)
    # __init__

    def get_data_by_key(self, key):
        """Helper to access a pretrained model data through a key."""
        assert key in self.mat_params_dict, "key: " + key + " not found."
        return self.mat_params_dict[key]
    # get_data_by_key

    def _weight_variable_on_cpu(self, name, shape):
        """Helper to create a weight Variable stored on CPU memory.

        Args:
          name: name of the variable.
          shape: list of ints: (height, width, channel, filter).

        Returns:
          initializer for Variable.
        """
        assert len(shape) == 4

        weights = self.get_data_by_key(name + "_filter")  # (h, w, channel, filter)
        assert list(weights.shape) == shape
        initializer = tf.constant_initializer(weights, dtype=self.dtype)

        # with tf.device('/cpu:0'):
        with tf.device(self.device_string):
            var = tf.get_variable(name + "_w", shape, initializer=initializer, dtype=self.dtype)
        # with
        return var
    # _weight_variable_on_cpu

    def _bias_variable_on_cpu(self, name, shape):
        """Helper to create a bias Variable stored on CPU memory.

        Args:
          name: name of the variable.
          shape: int, filter size.

        Returns:
          initializer for Variable.
        """
        assert isinstance(shape, int)
        bias = self.get_data_by_key(name + "_bias")
        assert len(bias) == shape
        initializer = tf.constant_initializer(bias, dtype=self.dtype)

        # with tf.device('/cpu:0'):
        with tf.device(self.device_string):
            var = tf.get_variable(name + "_b", shape, initializer=initializer, dtype=self.dtype)
        return var
    # _bias_variable_on_cpu

    def _bn_variable_on_cpu(self, name, shape):
        """Helper to create a batch normalization Variable stored on CPU memory.

        Args:
          name: name of the variable.
          shape: int, filter size.

        Returns:
          initializer for Variable.
        """
        assert isinstance(shape, int)

        name2 = "bn" + name[3:]
        if name.startswith("conv"):
            name2 = "bn_" + name

        scale = self.get_data_by_key(name2 + '_scale')
        offset = self.get_data_by_key(name2 + '_offset')
        mean = self.get_data_by_key(name2 + '_mean')
        variance = self.get_data_by_key(name2 + '_variance')

        # with tf.device('/cpu:0'):
        with tf.device(self.device_string):
            initializer = tf.constant_initializer(scale, dtype=self.dtype)
            scale = tf.get_variable(name2 + "_scale", shape, initializer=initializer, dtype=self.dtype)
            initializer = tf.constant_initializer(offset, dtype=self.dtype)
            offset = tf.get_variable(name2 + "_offset", shape, initializer=initializer, dtype=self.dtype)
            initializer = tf.constant_initializer(mean, dtype=self.dtype)
            mean = tf.get_variable(name2 + "_mean", shape, initializer=initializer, dtype=self.dtype)
            initializer = tf.constant_initializer(variance, dtype=self.dtype)
            variance = tf.get_variable(name2 + "_variance", shape, initializer=initializer, dtype=self.dtype)

        return scale, offset, mean, variance
    # _bn_variable_on_cpu

    def conv_block(self, bottom, name, shape, strides=[1,1,1,1], padding="SAME",
                   has_bias=False, add_relu=True, add_bn=True, eps=1.0e-5):
        """Create a block composed of multiple layers:
              a conv layer
              a batch normalization layer
              an activation layer

        Args:
          bottom: A layer before this block.
          name: Name of the block.
          shape: List of ints: (height, width, channel, filter).
          strides: Strides of conv layer.
          padding: Padding of conv layer.
          has_bias: Whether a bias term is added.
          add_relu: Whether a ReLU layer is added.
          add_bn: Whether a batch normalization layer is added.
          eps: A small float number to avoid dividing by 0, used in a batch normalization layer.
        Returns:
          a block of layers
        """
        assert len(shape) == 4

        weight = self._weight_variable_on_cpu(name, shape)
        conv = tf.nn.conv2d(bottom, weight, strides, padding=padding)
        if has_bias:
            bias = self._bias_variable_on_cpu(name, shape[3])

        pre_activation = tf.nn.bias_add(conv, bias) if has_bias else conv

        if add_bn:
            # scale, offset, mean, variance = self._bn_variable_on_cpu("bn_" + name, shape[-1])
            scale, offset, mean, variance = self._bn_variable_on_cpu(name, shape[-1])
            pre_activation = tf.nn.batch_normalization(pre_activation, mean, variance, offset, scale, variance_epsilon=eps)

        relu = tf.nn.relu(pre_activation) if add_relu else pre_activation

        return relu
    # conv_block


    def conv_trans_layer(self, bottom, name, shape, strides=[1,1,1,1], padding="SAME", has_bias=False):
        """Create a block composed of multiple layers:
              a transpose of conv layer
              an activation layer

        Args:
          bottom: A layer before this block.
          name: Name of the block.
          shape: List of ints: (height, width, channel, filter).
          strides: Strides of conv layer.
          padding: Padding of conv layer.
          has_bias: Whether a bias term is added.
          add_relu: Whether a ReLU layer is added.
        Returns:
          a block of layers
        """
        assert len(shape) == 4

        weight = self._weight_variable_on_cpu(name, shape)
        nb, h, w, nc = tf.split(tf.shape(bottom), num_or_size_splits=4)
        output_shape = tf.stack([nb, (h - 1) * strides[1] - 3 + shape[0], (w - 1) * strides[2] - 3 + shape[1], nc])[:, 0]
        conv = tf.nn.conv2d_transpose(bottom, weight, output_shape, strides, padding=padding)
        if has_bias:
            bias = self._bias_variable_on_cpu(name, shape[3])

        conv = tf.nn.bias_add(conv, bias) if has_bias else conv

        return conv
    # conv_trans_layer

    def residual_block(self, bottom, name, in_channel, neck_channel, out_channel, trunk):
        """Create a residual block.

        Args:
          bottom: A layer before this block.
          name: Name of the block.
          in_channel: number of channels in a input tensor.
          neck_channel: number of channels in a bottleneck block.
          out_channel: number of channels in a output tensor.
          trunk: a tensor in a identity path.
        Returns:
          a block of layers
        """
        _strides = [1, 2, 2, 1] if name.startswith("res3a") or name.startswith("res4a") else [1, 1, 1, 1]
        res = self.conv_block(bottom, name + '_branch2a', shape=[1, 1, in_channel, neck_channel],
                              strides=_strides, padding="VALID", add_relu=True)
        res = self.conv_block(res, name + '_branch2b', shape=[3, 3, neck_channel, neck_channel],
                              padding="SAME", add_relu=True)
        res = self.conv_block(res, name + '_branch2c', shape=[1, 1, neck_channel, out_channel],
                              padding="VALID", add_relu=False)

        res = trunk + res
        res = tf.nn.relu(res)

        return res
    # residual_block

    def tiny_face(self, image):
        """Create a tiny face model.

        Args:
          image: an input image.
        Returns:
          a score tensor
        """
        img = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        conv = self.conv_block(img, 'conv1', shape=[7, 7, 3, 64], strides=[1, 2, 2, 1], padding="VALID", add_relu=True)
        pool1 = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        res2a_branch1 = self.conv_block(pool1, 'res2a_branch1', shape=[1, 1, 64, 256], padding="VALID", add_relu=False)
        res2a = self.residual_block(pool1, 'res2a', 64, 64, 256, res2a_branch1)
        res2b = self.residual_block(res2a, 'res2b', 256, 64, 256, res2a)
        res2c = self.residual_block(res2b, 'res2c', 256, 64, 256, res2b)

        res3a_branch1 = self.conv_block(res2c, 'res3a_branch1', shape=[1, 1, 256, 512], strides=[1, 2, 2, 1], padding="VALID", add_relu=False)
        res3a = self.residual_block(res2c, 'res3a', 256, 128, 512, res3a_branch1)

        res3b1 = self.residual_block(res3a, 'res3b1', 512, 128, 512, res3a)
        res3b2 = self.residual_block(res3b1, 'res3b2', 512, 128, 512, res3b1)
        res3b3 = self.residual_block(res3b2, 'res3b3', 512, 128, 512, res3b2)

        res4a_branch1 = self.conv_block(res3b3, 'res4a_branch1', shape=[1, 1, 512, 1024], strides=[1, 2, 2, 1], padding="VALID", add_relu=False)
        res4a = self.residual_block(res3b3, 'res4a', 512, 256, 1024, res4a_branch1)

        res4b = res4a
        for i in range(1, 23):
            res4b = self.residual_block(res4b, 'res4b' + str(i), 1024, 256, 1024, res4b)

        score_res4 = self.conv_block(res4b, 'score_res4', shape=[1, 1, 1024, 125], padding="VALID",
                                     has_bias=True, add_relu=False, add_bn=False)
        score4 = self.conv_trans_layer(score_res4, 'score4', shape=[4, 4, 125, 125], strides=[1, 2, 2, 1], padding="SAME")
        score_res3 = self.conv_block(res3b3, 'score_res3', shape=[1, 1, 512, 125], padding="VALID",
                                     has_bias=True, add_bn=False, add_relu=False)

        bs, height, width = tf.split(tf.shape(score4), num_or_size_splits=4)[0:3]
        _size = tf.convert_to_tensor([height[0], width[0]])
        _offsets = tf.zeros([bs[0], 2])
        score_res3c = tf.image.extract_glimpse(score_res3, _size, _offsets, centered=True, normalized=False)

        score_final = score4 + score_res3c
        return score_final
    # tiny_face
# Model

def matconvnet_hr101_to_pickle(matlab_model_path, weight_file_path):
    """
    Input:
        + matlab_model_path: Matlab pretrained model, /path/to/hr_res101.mat
        + weight_file_path: Weight file for Tensorflow, /path/to/mat2tf.pkl
    """
    # check arguments
    assert os.path.exists(matlab_model_path), \
        "Matlab pretrained model: " + matlab_model_path + " not found."
    assert os.path.exists(os.path.dirname((weight_file_path))),\
        "Directory for weight file for Tensorflow: " + weight_file_path + " not found."

    mat_params_dict = {}
    mat_blocks_dict = {}

    f = sio.loadmat(matlab_model_path)
    net = f['net']
    clusters = np.copy(net['meta'][0][0][0][0][6])
    average_image = np.copy(net['meta'][0][0][0][0][2][0][0][2])[:, 0]
    mat_params_dict["clusters"] = clusters
    mat_params_dict["average_image"] = average_image

    layers = net['layers'][0][0][0]
    mat_params = net['params'][0][0][0]
    for p in mat_params:
        mat_params_dict[p[0][0]] = p[1]

    for k, layer in enumerate(layers):
        type_string = ''
        param_string = ''

        layer_name, layer_type = layer[0][0], layer[1][0]
        layer_inputs = []
        layer_outputs = []
        layer_params = []

        layer_inputs_count = layer[2][0].shape[0]
        for i in range(layer_inputs_count):
            layer_inputs.append(layer[2][0][i][0])

        layer_outputs_count = layer[3][0].shape[0]
        for i in range(layer_outputs_count):
            layer_outputs.append(layer[3][0][i][0])

        if layer[4].shape[0] > 0:
            layer_params_count = layer[4][0].shape[0]
            for i in range(layer_params_count):
                layer_params.append(layer[4][0][i][0])

        mat_blocks_dict[layer_name + '_type'] = layer_type
        mat_params_dict[layer_name + '_type'] = layer_type
        if layer_type == u'dagnn.Conv':
            nchw = layer[5][0][0][0][0]
            has_bias = layer[5][0][0][1][0][0]
            pad = layer[5][0][0][3][0]
            stride = layer[5][0][0][4][0]
            dilate = layer[5][0][0][5][0]
            mat_blocks_dict[layer_name + '_nchw'] = nchw
            mat_blocks_dict[layer_name + '_has_bias'] = has_bias
            mat_blocks_dict[layer_name + '_pad'] = pad
            mat_blocks_dict[layer_name + '_stride'] = stride
            mat_blocks_dict[layer_name + '_dilate'] = dilate
            if has_bias:
                bias = mat_params_dict[layer_name + '_bias'][0] # (1, N) -> (N,)
                mat_params_dict[layer_name + '_bias'] = bias
        elif layer_type == u'dagnn.BatchNorm':
            epsilon = layer[5][0][0][1][0][0]
            gamma = mat_params_dict[layer_name + '_mult'][:, 0] # (N, 1) -> (N,)
            beta = mat_params_dict[layer_name + '_bias'][:, 0] # (N, 1) -> (N,)
            moments = mat_params_dict[layer_name + '_moments'] # (N, 2) -> (N,), (N,)
            moving_mean = moments[:, 0]
            moving_var = moments[:, 1] * moments[:, 1] - epsilon

            mat_blocks_dict[layer_name + '_variance_epsilon'] = epsilon
            mat_params_dict[layer_name + '_scale'] = gamma
            mat_params_dict[layer_name + '_offset'] = beta
            mat_params_dict[layer_name + '_mean'] = moving_mean
            mat_params_dict[layer_name + '_variance'] = moving_var
        elif layer_type == u'dagnn.ConvTranspose':
            nchw = layer[5][0][0][0][0]
            has_bias = layer[5][0][0][1][0][0]
            upsample = layer[5][0][0][2][0]
            crop = layer[5][0][0][3][0]
            mat_blocks_dict[layer_name + '_nchw'] = nchw
            mat_blocks_dict[layer_name + '_has_bias'] = has_bias
            mat_blocks_dict[layer_name + '_upsample'] = upsample
            mat_blocks_dict[layer_name + '_crop'] = crop
            wmat = mat_params_dict[layer_name + 'f']
            mat_params_dict[layer_name + '_filter'] = wmat
        elif layer_type == u'dagnn.Pooling':
            method = layer[5][0][0][0][0]
            pool_size = layer[5][0][0][1][0]
            pad = layer[5][0][0][3][0]
            stride = layer[5][0][0][4][0]
            mat_blocks_dict[layer_name + '_method'] = method
            mat_blocks_dict[layer_name + '_pool_size'] = pool_size
            mat_blocks_dict[layer_name + '_pad'] = pad
            mat_blocks_dict[layer_name + '_stride'] = stride
        elif layer_type == u'dagnn.ReLU':
            pass
        elif layer_type == u'dagnn.Sum':
            pass
        else:
            pass
        # if
    # for

    with open(weight_file_path, 'wb') as f:
        pickle.dump([mat_blocks_dict, mat_params_dict], f, pickle.HIGHEST_PROTOCOL)
    # with
# matconvnet_hr101_to_pickle

def main():
    argparse = ArgumentParser()
    argparse.add_argument('--weight_file_path', type=str, help='Pretrained weight file.', default="/path/to/mat2tf.pkl")
    argparse.add_argument('--data_dir', type=str, help='Image data directory.', default="/path/to/input_image_directory")
    argparse.add_argument('--output_dir', type=str, help='Output directory for images with faces detected.', default="/path/to/output_directory")
    argparse.add_argument('--prob_thresh', type=float, help='The threshold of detection confidence(default: 0.5).', default=0.5)
    argparse.add_argument('--nms_thresh', type=float, help='The overlap threshold of non maximum suppression(default: 0.1).', default=0.1)
    argparse.add_argument('--line_width', type=int, help='Line width of bounding boxes(0: auto).', default=3)
    argparse.add_argument('--display', type=bool, help='Display each image on window.', default=False)

    args = argparse.parse_args()

    # check arguments
    assert os.path.exists(args.weight_file_path), "weight file: " + args.weight_file_path + " not found."
    assert os.path.exists(args.data_dir), "data directory: " + args.data_dir + " not found."
    assert os.path.exists(args.output_dir), "output directory: " + args.output_dir + " not found."
    assert args.line_width >= 0, "line_width should be >= 0."

    with tf.Graph().as_default():
        evaluate(
          weight_file_path=args.weight_file_path, data_dir=args.data_dir, output_dir=args.output_dir,
          prob_thresh=args.prob_thresh, nms_thresh=args.nms_thresh,
          lw=args.line_width, display=args.display)
# main