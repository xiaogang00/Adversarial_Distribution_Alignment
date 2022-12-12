import os
import time
import logging
import argparse

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.nn as nn

from models.pspnet import PSPNet, DeepLabV3
import data.dataset as dataset
import data.transform as transform
from util_file import AverageMeter, intersectionAndUnion, check_makedirs, colorize
from options.test_options import TestOptions
import models.networks as networks
import torch.nn.functional as F

cv2.ocl.setUseOpenCL(False)


def FGSM(input, target, model, clip_min, clip_max, eps=0.2, zoom_factor=8):
    input_variable = input.detach().clone()
    input_variable.requires_grad = True
    model.zero_grad()
    result = model(input_variable)
    if zoom_factor != 8:
        h = int((target.size()[1] - 1) / 8 * zoom_factor + 1)
        w = int((target.size()[2] - 1) / 8 * zoom_factor + 1)
        # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
        target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='bilinear', align_corners=True).squeeze(1).long()

    ignore_label = 255
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_label).cuda()
    loss = criterion(result, target.detach())
    loss.backward()
    res = input_variable.grad

    ################################################################################
    adversarial_example = input.detach().clone()
    adversarial_example[:, 0, :, :] = adversarial_example[:, 0, :, :] * std_origin[0] + mean_origin[0]
    adversarial_example[:, 1, :, :] = adversarial_example[:, 1, :, :] * std_origin[1] + mean_origin[1]
    adversarial_example[:, 2, :, :] = adversarial_example[:, 2, :, :] * std_origin[2] + mean_origin[2]
    adversarial_example = adversarial_example + eps * torch.sign(res)
    adversarial_example = torch.max(adversarial_example, clip_min)
    adversarial_example = torch.min(adversarial_example, clip_max)
    adversarial_example = torch.clamp(adversarial_example, min=0.0, max=1.0)

    adversarial_example[:, 0, :, :] = (adversarial_example[:, 0, :, :] - mean_origin[0]) / std_origin[0]
    adversarial_example[:, 1, :, :] = (adversarial_example[:, 1, :, :] - mean_origin[1]) / std_origin[1]
    adversarial_example[:, 2, :, :] = (adversarial_example[:, 2, :, :] - mean_origin[2]) / std_origin[2]
    ################################################################################
    return adversarial_example


def BIM_label2(input, target, model, eps=0.03, k_number=3, alpha=0.01):
    input_unnorm = input.clone().detach()
    input_unnorm[:, 0, :, :] = input_unnorm[:, 0, :, :] * std_origin[0] + mean_origin[0]
    input_unnorm[:, 1, :, :] = input_unnorm[:, 1, :, :] * std_origin[1] + mean_origin[1]
    input_unnorm[:, 2, :, :] = input_unnorm[:, 2, :, :] * std_origin[2] + mean_origin[2]
    clip_min = input_unnorm - eps
    clip_max = input_unnorm + eps

    adversarial_example = input.detach().clone()
    adversarial_example.requires_grad = True
    for mm in range(k_number):
        adversarial_example = FGSM(adversarial_example, target, model, clip_min, clip_max, eps=alpha)
        adversarial_example = adversarial_example.detach()
        adversarial_example.requires_grad = True
        model.zero_grad()
    return adversarial_example


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def main():
    global logger
    logger = get_logger()

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    global mean_origin
    global std_origin
    mean_origin = [0.485, 0.456, 0.406]
    std_origin = [0.229, 0.224, 0.225]

    global mean_generation
    global std_generation
    mean_generation = [0.5, 0.5, 0.5]
    std_generation = [0.5, 0.5, 0.5]

    save_folder = 'checkpoints/voc_model_pspnet/results'
    save_folder_list = [save_folder]

    save_path = 'checkpoints/voc_model_pspnet/last_net_G.pth'
    save_path_list = [save_path]

    test_transform = transform.Compose([transform.ToTensor()])
    split = 'val'
    data_root = '/mnt/backup/project/hszhao/dataset/voc2012'
    test_list = '/mnt/backup/project/hszhao/dataset/voc2012/list/val.txt'
    test_data = dataset.SemData(split=split, data_root=data_root, data_list=test_list, transform=test_transform)
    index_start = 0
    index_end = len(test_data.data_list)
    test_data.data_list = test_data.data_list[index_start:index_end]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    colors_path = '/mnt/proj3/xgxu/semseg_mp/dataset/voc2012/voc2012_colors.txt'
    names_path = '/mnt/proj3/xgxu/semseg_mp/dataset/voc2012/voc2012_names.txt'
    colors = np.loadtxt(colors_path).astype('uint8')
    names = [line.rstrip('\n') for line in open(names_path)]

    model_black = DeepLabV3(layers=50, classes=21, zoom_factor=8, pretrained=False)
    model_black = torch.nn.DataParallel(model_black).cuda()
    cudnn.benchmark = True
    model_path = 'model_seg_NoD/voc2012/deeplab/train_epoch_50.pth'
    checkpoint = torch.load(model_path)
    model_black.load_state_dict(checkpoint['state_dict'], strict=False)
    model_black.eval()
    print('loading successfully')

    model = PSPNet(layers=50, classes=21, zoom_factor=8, pretrained=False)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    model_path = 'model_seg_NoD/voc2012/pspnet/train_epoch_50.pth'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print('loading successfully')

    opt = TestOptions().parse(save=False)
    netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                             opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                             opt.n_blocks_local, opt.norm, gpu_ids=opt.gpu_ids)
    global test_h
    global test_w
    test_h = 417
    test_w = 417

    for mm in range(len(save_folder_list)):
        save_folder = save_folder_list[mm]
        if not (os.path.exists(save_folder)):
            os.mkdir(save_folder)
        gray_folder = os.path.join(save_folder, 'gray')
        color_folder = os.path.join(save_folder, 'color')
        save_path = save_path_list[mm]
        netG.load_state_dict(torch.load(save_path))

        classes = 21
        base_size = 512
        scales = [1.0]
        test(test_loader, test_data.data_list, model, model_black, netG, classes, mean, std, base_size, test_h, test_w,
             scales, gray_folder, color_folder, colors)
        cal_acc(test_data.data_list, gray_folder, classes, names)


def net_process(model, model_black, net_G, image, target, mean, std=None, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    target = torch.from_numpy(target).long()

    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()
    target = target.unsqueeze(0).cuda()

    if flip:
        input = torch.cat([input, input.flip(3)], 0)
        target = torch.cat([target, target.flip(2)], 0)

    adver_input = BIM_label2(input, target, model_black, eps=0.03, k_number=1, alpha=0.01)
    with torch.no_grad():

        adver_input[:, 0, :, :] = adver_input[:, 0, :, :] * std_origin[0] + mean_origin[0]
        adver_input[:, 1, :, :] = adver_input[:, 1, :, :] * std_origin[1] + mean_origin[1]
        adver_input[:, 2, :, :] = adver_input[:, 2, :, :] * std_origin[2] + mean_origin[2]
        adver_input[:, 0, :, :] = (adver_input[:, 0, :, :] - mean_generation[0]) / std_generation[0]
        adver_input[:, 1, :, :] = (adver_input[:, 1, :, :] - mean_generation[1]) / std_generation[1]
        adver_input[:, 2, :, :] = (adver_input[:, 2, :, :] - mean_generation[2]) / std_generation[2]
        adver_input = F.interpolate(adver_input, size=(512, 512), mode='bilinear', align_corners=True)

        adver_input = net_G.forward(adver_input)

        adver_input = F.interpolate(adver_input, size=(test_h, test_w), mode='bilinear', align_corners=True)
        adver_input[:, 0, :, :] = adver_input[:, 0, :, :] * std_generation[0] + mean_generation[0]
        adver_input[:, 1, :, :] = adver_input[:, 1, :, :] * std_generation[1] + mean_generation[1]
        adver_input[:, 2, :, :] = adver_input[:, 2, :, :] * std_generation[2] + mean_generation[2]
        adver_input[:, 0, :, :] = (adver_input[:, 0, :, :] - mean_origin[0]) / std_origin[0]
        adver_input[:, 1, :, :] = (adver_input[:, 1, :, :] - mean_origin[1]) / std_origin[1]
        adver_input[:, 2, :, :] = (adver_input[:, 2, :, :] - mean_origin[2]) / std_origin[2]
        output = model(adver_input)

    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, model_black, net_G, image, target, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
        target = cv2.copyMakeBorder(target, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=255)

    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()

            target_crop = target[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, model_black, net_G, image_crop, target_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def test(test_loader, data_list, model, model_black, net_G, classes, mean, std, base_size,
         crop_h, crop_w, scales, gray_folder, color_folder, colors):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):

        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        color_path = os.path.join(color_folder, image_name + '.png')

        data_time.update(time.time() - end)
        input = np.squeeze(input.numpy(), axis=0)
        target = np.squeeze(target.numpy(), axis=0)
        image = np.transpose(input, (1, 2, 0))

        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)

            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            target_scale = cv2.resize(target.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            prediction += scale_process(model, model_black, net_G, image_scale, target_scale, classes, crop_h, crop_w, h, w, mean, std)
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        check_makedirs(gray_folder)
        check_makedirs(color_folder)
        gray = np.uint8(prediction)
        color = colorize(gray, colors)
        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')
        cv2.imwrite(gray_path, gray)
        color.save(color_path)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def cal_acc(data_list, pred_folder, classes, names):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        # target = cv2.resize(target, (1024, 512), interpolation=cv2.INTER_NEAREST)

        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        logger.info('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name+'.png', accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))


if __name__ == '__main__':
    main()
