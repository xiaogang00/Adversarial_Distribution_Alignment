"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

VOC_ROOT = '/mnt/proj3/xgxu/SSD/test/'
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects

def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath.format(set_type), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap



from RFB.utils.box_utils import match, log_sum_exp
import torch.nn.functional as F
def attack_loc(x, targets, priors, net, threshold=0.5,  num_classes=21, eps=8.0, variance=[0.1, 0.2]):
    input_x = x.clone().detach()
    input_x.requires_grad = True

    predictions = net(input_x, indicate=1)
    loc_data, conf_data = predictions
    priors = priors
    num = loc_data.size(0)
    num_priors = (priors.size(0))

    loc_t = torch.Tensor(num, num_priors, 4)
    conf_t = torch.LongTensor(num, num_priors)
    for idx in range(num):
        truths = targets[idx][:, :-1].data
        labels = targets[idx][:, -1].data
        defaults = priors.data
        match(threshold, truths, defaults, variance, labels, loc_t, conf_t, idx)
    loc_t = loc_t.cuda()
    conf_t = conf_t.cuda()
    # wrap targets
    loc_t = Variable(loc_t, requires_grad=False)
    conf_t = Variable(conf_t, requires_grad=False)

    pos = conf_t > 0
    pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
    loc_p = loc_data[pos_idx].view(-1, 4)
    loc_t = loc_t[pos_idx].view(-1, 4)
    loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

    pos = pos.view(-1, 1)
    num_pos = pos.long().sum(1, keepdim=True)
    N = max(num_pos.data.sum().float(), 1)
    loss_l /= N
    loss_l.backward()
    grad_value = input_x.grad
    adversarial_example = input_x + eps * torch.sign(grad_value)
    return adversarial_example


def attack_cls(x, targets, priors, net, threshold=0.5, num_classes=21, eps=8.0, negpos_ratio=3, variance=[0.1, 0.2]):
    input_x = x.clone().detach()
    input_x.requires_grad = True

    predictions = net(input_x, indicate=1)
    loc_data, conf_data = predictions
    priors = priors
    num = loc_data.size(0)
    num_priors = (priors.size(0))

    # match priors (default boxes) and ground truth boxes
    loc_t = torch.Tensor(num, num_priors, 4)
    conf_t = torch.LongTensor(num, num_priors)
    for idx in range(num):
        truths = targets[idx][:, :-1].data
        labels = targets[idx][:, -1].data
        defaults = priors.data
        match(threshold, truths, defaults, variance, labels, loc_t, conf_t, idx)
    conf_t = conf_t.cuda()
    conf_t = Variable(conf_t, requires_grad=False)

    pos = conf_t > 0
    # Compute max conf across batch for hard negative mining
    batch_conf = conf_data.view(-1, num_classes)
    loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

    # Hard Negative Mining
    pos = pos.view(-1, 1)
    loss_c[pos] = 0  # filter out pos boxes for now
    loss_c = loss_c.view(num, -1)
    _, loss_idx = loss_c.sort(1, descending=True)
    _, idx_rank = loss_idx.sort(1)
    num_pos = pos.long().sum(1, keepdim=True)
    num_neg = torch.clamp(negpos_ratio * num_pos, max=pos.size(1) - 1)
    idx_rank = idx_rank.view(-1, 1)
    neg = idx_rank < num_neg.expand_as(idx_rank)

    # Confidence Loss Including Positive and Negative Examples
    pos = pos.view(conf_data.shape[0], conf_data.shape[1])
    neg = neg.view(conf_data.shape[0], conf_data.shape[1])
    pos_idx = pos.unsqueeze(2).expand_as(conf_data)
    neg_idx = neg.unsqueeze(2).expand_as(conf_data)
    conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, num_classes)
    targets_weighted = conf_t[(pos + neg).gt(0)]

    if conf_p.shape[0] == 0:
        return input_x
    else:
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

    N = max(num_pos.data.sum().float(), 1)
    loss_c /= N
    loss_c.backward()
    grad_value = input_x.grad
    adversarial_example = input_x + eps * torch.sign(grad_value)
    return adversarial_example


def attack_cls_loc(x, targets, priors, net, threshold=0.5, num_classes=21, eps=8.0, negpos_ratio=3, variance=[0.1, 0.2]):
    input_x = x.clone().detach()
    input_x.requires_grad = True

    predictions = net(input_x, indicate=1)
    loc_data, conf_data = predictions
    priors = priors
    num = loc_data.size(0)
    num_priors = (priors.size(0))

    # match priors (default boxes) and ground truth boxes
    loc_t = torch.Tensor(num, num_priors, 4)
    conf_t = torch.LongTensor(num, num_priors)
    for idx in range(num):
        truths = targets[idx][:, :-1].data
        labels = targets[idx][:, -1].data
        defaults = priors.data
        match(threshold, truths, defaults, variance, labels, loc_t, conf_t, idx)
    conf_t = conf_t.cuda()
    loc_t = loc_t.cuda()
    conf_t = Variable(conf_t, requires_grad=False)
    loc_t = Variable(loc_t, requires_grad=False)

    pos = conf_t > 0
    pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
    loc_p = loc_data[pos_idx].view(-1, 4)
    loc_t = loc_t[pos_idx].view(-1, 4)
    loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

    # Compute max conf across batch for hard negative mining
    batch_conf = conf_data.view(-1, num_classes)
    loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

    # Hard Negative Mining
    pos = pos.view(-1, 1)
    loss_c[pos] = 0  # filter out pos boxes for now
    loss_c = loss_c.view(num, -1)
    _, loss_idx = loss_c.sort(1, descending=True)
    _, idx_rank = loss_idx.sort(1)
    num_pos = pos.long().sum(1, keepdim=True)
    num_neg = torch.clamp(negpos_ratio * num_pos, max=pos.size(1) - 1)
    idx_rank = idx_rank.view(-1, 1)
    neg = idx_rank < num_neg.expand_as(idx_rank)

    # Confidence Loss Including Positive and Negative Examples
    pos = pos.view(conf_data.shape[0], conf_data.shape[1])
    neg = neg.view(conf_data.shape[0], conf_data.shape[1])
    pos_idx = pos.unsqueeze(2).expand_as(conf_data)
    neg_idx = neg.unsqueeze(2).expand_as(conf_data)
    conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, num_classes)
    targets_weighted = conf_t[(pos + neg).gt(0)]

    if conf_p.shape[0] == 0:
        loss_c = 0
    else:
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

    N = num_pos.data.sum()
    loss_c /= N
    loss_l /= N
    loss_a = loss_c + loss_l
    loss_a.backward()
    grad_value = input_x.grad
    adversarial_example = input_x + eps * torch.sign(grad_value)
    return adversarial_example



def test_net(save_folder, net, net_black, netG, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()

        x = x[:, [2, 1, 0], :, :]
        scale = torch.Tensor([im.shape[1], im.shape[0],
                              im.shape[1], im.shape[0]])
        scale = scale.cuda()
        gt_input = torch.Tensor(gt).cuda()
        scale_this = scale.view(1, 4).repeat(gt_input.shape[0], 1)
        x = attack_cls_loc(x, Variable(gt_input.unsqueeze(0)), priors, net_black)
        x = x[:, [2, 1, 0], :, :]

        h_this = x.shape[2]
        w_this = x.shape[3]
        x[:, 0, :, :] = x[:, 0, :, :] + dataset_mean[2]
        x[:, 1, :, :] = x[:, 1, :, :] + dataset_mean[1]
        x[:, 2, :, :] = x[:, 2, :, :] + dataset_mean[0]
        x = x * 1.0 / 255.0
        x = 2 * x - 1
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        x = netG.forward(x)
        x = F.interpolate(x, size=(h_this, w_this), mode='bilinear', align_corners=True)
        x = (x + 1) * 0.5
        x = x * 255.0
        x[:, 0, :, :] = x[:, 0, :, :] - dataset_mean[2]
        x[:, 1, :, :] = x[:, 1, :, :] - dataset_mean[1]
        x[:, 2, :, :] = x[:, 2, :, :] - dataset_mean[0]

        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)

from RFB.data import VOC_300
from RFB.layers.functions import PriorBox
cfg = VOC_300
VOCroot = '/mnt/proj3/xgxu/SSD/test/'
priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


if __name__ == '__main__':
    # load net
    from model import networks
    num_classes = len(labelmap) + 1                      # +1 for background
    net = build_ssd('test', 300, num_classes)            # initialize SSD
    model_path = '../model_det_NoD/ssd/ssd_300_VOC0712.pth'
    net.load_state_dict(torch.load(model_path))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = VOCDetection(args.voc_root, [('2007', set_type)],
                           BaseTransform(300, dataset_mean),
                           VOCAnnotationTransform())

    input_nc = 3
    output_nc = 3
    ngf = 64
    netG = 'global'
    n_downsample_global = 4
    n_blocks_global = 9
    n_local_enhancers = 1
    n_blocks_local = 3
    norm = 'instance'
    gpu_ids = [0]
    netG = networks.define_G(input_nc, output_nc, ngf, netG,
                             n_downsample_global, n_blocks_global, n_local_enhancers,
                             n_blocks_local, norm, gpu_ids=gpu_ids)
    save_path = '../checkpoints/voc0712_model_ssd//last_net_G.pth'
    save_path_list = [save_path]

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    from RFB.models.RFB_Net_vgg import build_net
    img_dim = 300
    num_classes = 21
    net_black = build_net('test', img_dim, num_classes)  # initialize detector
    model_black_path = '../model_det_NoD/rbfnet/RFBNet300_VOC_80_7.pth'
    state_dict = torch.load(model_black_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net_black.load_state_dict(new_state_dict)
    net_black.eval()
    print('Finished loading model!')
    print(net)
    net_black = net_black.cuda()
    cudnn.benchmark = True

    for mm in range(len(save_path_list)):
        netG.load_state_dict(torch.load(save_path_list[mm]))
        netG.eval()
        # evaluation

        test_net(args.save_folder, net, net_black, netG, args.cuda, dataset,
                 BaseTransform(net.size, dataset_mean), args.top_k, 300,
                 thresh=args.confidence_threshold)
