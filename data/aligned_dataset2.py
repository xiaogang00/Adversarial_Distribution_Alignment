import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from PIL import Image
from . import transform
import cv2
import numpy as np
import torch

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        data_root = '/mnt/backup/project/hszhao/dataset/voc2012'
        split = 'train'
        data_list = '/mnt/backup/project/hszhao/dataset/voc2012/list/train_aug.txt'
        self.data_list = make_dataset(split, data_root, data_list)

        scale_min = 0.5
        scale_max = 2.0
        rotate_min = -10
        rotate_max = 10
        ignore_label = 255
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]
        train_h = 417
        train_w = 417

        train_transform = transform.Compose([
            transform.RandScale([scale_min, scale_max]),
            transform.RandRotate([rotate_min, rotate_max], padding=mean, ignore_label=ignore_label),
            transform.RandomGaussianBlur(),
            transform.RandomHorizontalFlip(),
            transform.Crop([train_h, train_w], crop_type='rand', padding=mean, ignore_label=ignore_label),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
        self.transform = train_transform


    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        image, label = self.transform(image, label)

        inst_tensor = torch.zeros_like(image)
        feat_tensor = torch.zeros_like(image)
        input_dict = {'label': label, 'inst': inst_tensor, 'image': image, 'path': image_path, 'feat': feat_tensor}
        return input_dict

    def __len__(self):
        return len(self.data_list)

    def name(self):
        return 'AlignedDataset'

