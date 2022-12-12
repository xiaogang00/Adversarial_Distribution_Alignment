import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

from . import pspnet
import torch.nn as nn
import random
import torch.nn.functional as F
import random
import cv2
from .detection.layers.box_utils import match, log_sum_exp
from .detection.layers.modules import MultiBoxLoss

def attack_loc(x, targets, net, threshold=0.5,  num_classes=21, eps=8.0, variance=[0.1, 0.2]):
    input_x = x.clone().detach()
    input_x.requires_grad = True

    predictions = net(input_x, indicate=0)
    loc_data, conf_data, priors = predictions
    num = loc_data.size(0)
    priors = priors[:loc_data.size(1), :]
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
    N = num_pos.data.sum()
    loss_l /= N
    loss_l.backward()
    grad_value = input_x.grad
    adversarial_example = input_x + eps * torch.sign(grad_value)
    return adversarial_example


def attack_cls(x, targets, net, threshold=0.5, num_classes=21, eps=8.0, negpos_ratio=3, variance=[0.1, 0.2]):
    input_x = x.clone().detach()
    input_x.requires_grad = True

    predictions = net(input_x, indicate=0)
    loc_data, conf_data, priors = predictions
    num = loc_data.size(0)
    priors = priors[:loc_data.size(1), :]
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
    # print(conf_data.shape)
    pos = pos.view(conf_data.shape[0], conf_data.shape[1])
    neg = neg.view(conf_data.shape[0], conf_data.shape[1])
    pos_idx = pos.unsqueeze(2).expand_as(conf_data)
    neg_idx = neg.unsqueeze(2).expand_as(conf_data)
    conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, num_classes)
    targets_weighted = conf_t[(pos + neg).gt(0)]
    loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
    N = num_pos.data.sum()
    loss_c /= N
    loss_c.backward()
    grad_value = input_x.grad
    adversarial_example = input_x + eps * torch.sign(grad_value)
    return adversarial_example


class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, True, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, g_rec, g_seg1, g_seg2, d_real, d_fake):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, g_rec, g_seg1, g_seg2, d_real, d_fake), flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = False
        self.gen_features = False
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc        
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = opt.output_nc
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)

        ################################################################################################
        self.mean_origin = [0.485, 0.456, 0.406]
        self.std_origin = [0.229, 0.224, 0.225]
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        from .detection.ssd import build_ssd
        labelmap = (  # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')

        num_classes = len(labelmap) + 1  # +1 for background
        self.num_classes = num_classes
        detection_net = build_ssd('test', 300, num_classes)  # initialize SSD
        trained_model = 'model_det_NoD/ssd/ssd_300_VOC0712.pth'
        detection_net.load_state_dict(torch.load(trained_model))
        detection_net.eval()
        self.detection_net = detection_net.cuda()
        # self.dataset_mean = (104, 117, 123)
        self.dataset_mean = (123, 117, 104)
        self.detection_criterion = MultiBoxLoss(21, 0.5, True, 0, True, 3, 0.5, False, True)
        ################################################################################################

        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','G_rec', 'G_seg1', 'G_seg2', 'D_real', 'D_fake')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())         
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input3(self, label_map, inst_map=None, real_image=None, feat_map=None, obj_num=None, infer=False):

        target_label = []
        batch_size = label_map.shape[0]
        for mm in range(batch_size):
            obj_num_this = obj_num[mm]
            gt_this = Variable(label_map[mm][0:obj_num_this].cuda(), volatile=True)
            target_label.append(gt_this)

        real_image = Variable(real_image.data.cuda())
        real_image_clone = real_image.clone()
        eps = 8

        self.input_h = real_image.shape[2]
        self.input_w = real_image.shape[3]
        real_image[:, 0, :, :] = real_image[:, 0, :, :] - self.dataset_mean[0]
        real_image[:, 1, :, :] = real_image[:, 1, :, :] - self.dataset_mean[1]
        real_image[:, 2, :, :] = real_image[:, 2, :, :] - self.dataset_mean[2]
        if random.randint(0, 1) == 0:
            real_image = attack_cls(real_image, target_label, self.detection_net, eps=eps)
        else:
            real_image = attack_loc(real_image, target_label, self.detection_net, eps=eps)
        real_image[:, 0, :, :] = real_image[:, 0, :, :] + self.dataset_mean[0]
        real_image[:, 1, :, :] = real_image[:, 1, :, :] + self.dataset_mean[1]
        real_image[:, 2, :, :] = real_image[:, 2, :, :] + self.dataset_mean[2]

        real_image = real_image * 1.0 / 255.0
        real_image[:, 0, :, :] = (real_image[:, 0, :, :] - self.mean[0]) / self.std[0]
        real_image[:, 1, :, :] = (real_image[:, 1, :, :] - self.mean[1]) / self.std[1]
        real_image[:, 2, :, :] = (real_image[:, 2, :, :] - self.mean[2]) / self.std[2]
        real_image = F.interpolate(real_image, size=(256, 256), mode='bilinear', align_corners=True)

        real_image_clone = real_image_clone * 1.0 / 255.0
        real_image_clone[:, 0, :, :] = (real_image_clone[:, 0, :, :] - self.mean[0]) / self.std[0]
        real_image_clone[:, 1, :, :] = (real_image_clone[:, 1, :, :] - self.mean[1]) / self.std[1]
        real_image_clone[:, 2, :, :] = (real_image_clone[:, 2, :, :] - self.mean[2]) / self.std[2]
        real_image_clone = F.interpolate(real_image_clone, size=(256, 256), mode='bilinear', align_corners=True)

        input_label = target_label
        return input_label, inst_map, real_image_clone, feat_map, real_image, target_label

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = test_image.detach()
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, inst, image, feat, obj_num, infer=False):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map, adversarial_image, target_label = self.encode_input3(label, inst, image, feat, obj_num)

        # print(target_label.shape, real_image.shape)
        fake_image = self.netG.forward(adversarial_image)
        fake_image2 = self.netG.forward(real_image)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate(input_label, fake_image2)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(fake_image)
        loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, fake_image2) * self.opt.lambda_feat
            loss_G_VGG += self.criterionVGG(fake_image2, real_image) * self.opt.lambda_feat

        loss_G_rec = torch.mean(torch.abs(fake_image - fake_image2)) * 50
        loss_G_rec += torch.mean(torch.abs(fake_image2 - real_image)) * 50

        fake_image = F.interpolate(fake_image, size=(self.input_h, self.input_w), mode='bilinear', align_corners=True)
        std_map1 = torch.ones_like(fake_image[:, 0:1, :, :]).cuda() * self.std[0]
        std_map2 = torch.ones_like(fake_image[:, 0:1, :, :]).cuda() * self.std[1]
        std_map3 = torch.ones_like(fake_image[:, 0:1, :, :]).cuda() * self.std[2]
        mean_map1 = torch.ones_like(fake_image[:, 0:1, :, :]).cuda() * self.mean[0]
        mean_map2 = torch.ones_like(fake_image[:, 0:1, :, :]).cuda() * self.mean[1]
        mean_map3 = torch.ones_like(fake_image[:, 0:1, :, :]).cuda() * self.mean[2]
        std_map = torch.cat([std_map1, std_map2, std_map3], dim=1)
        mean_map = torch.cat([mean_map1, mean_map2, mean_map3], dim=1)

        mean_data_map1 = torch.ones_like(fake_image[:, 0:1, :, :]).cuda() * self.dataset_mean[0]
        mean_data_map2 = torch.ones_like(fake_image[:, 0:1, :, :]).cuda() * self.dataset_mean[1]
        mean_data_map3 = torch.ones_like(fake_image[:, 0:1, :, :]).cuda() * self.dataset_mean[2]
        mean_data_map = torch.cat([mean_data_map1, mean_data_map2, mean_data_map3], dim=1)

        fake_image = fake_image.mul(std_map).add(mean_map)
        fake_image = fake_image * 255.0
        fake_image = fake_image - mean_data_map

        result, feature1 = self.detection_net(fake_image, indicate=1)
        loss_l, loss_c = self.detection_criterion(result, target_label)
        loss_G_seg1 = loss_l + loss_c

        fake_image2 = F.interpolate(fake_image2, size=(self.input_h, self.input_w), mode='bilinear', align_corners=True)
        fake_image2 = fake_image2.mul(std_map).add(mean_map)
        fake_image2 = fake_image2 * 255.0
        fake_image2 = fake_image2 - mean_data_map

        result2, feature2 = self.detection_net(fake_image2, indicate=1)
        loss_l, loss_c = self.detection_criterion(result2, target_label)
        loss_G_seg1 += loss_l + loss_c

        real_image = F.interpolate(real_image, size=(self.input_h, self.input_w), mode='bilinear', align_corners=True)
        real_image = real_image.mul(std_map).add(mean_map)
        real_image = real_image * 255.0
        real_image = real_image - mean_data_map

        result_real, feature3 = self.detection_net(real_image, indicate=1)
        loss_G_seg1 += torch.mean(torch.abs(feature1 - feature3)) * 4
        loss_G_seg1 += torch.mean(torch.abs(feature2 - feature3)) * 4

        loss_cluster = 0
        loss_cluster2 = 0
        loss_class = 0
        cluster_class_real = []
        cluster_class_fake = []
        cluster_class_real1 = []
        cluster_class_fake1 = []
        cluster_class_real_num = []
        cluster_class_fake_num = []
        for mm in range(self.num_classes):
            cluster_class_real1.append(0)
            cluster_class_fake1.append(0)
            cluster_class_real_num.append(0)
            cluster_class_fake_num.append(0)

        f_h = feature1.shape[2]
        f_w = feature1.shape[3]
        batch_size = len(target_label)
        for mm in range(batch_size):
            obj_num = target_label[mm].shape[0]
            for nn in range(obj_num):
                box = target_label[mm][nn][0:4] * f_h
                label = target_label[mm][nn][4]
                x1 = max(0, int(box[0]))
                y1 = max(0, int(box[1]))
                x2 = min(f_w-1, int(box[2]))
                y2 = min(f_h-1, int(box[3]))
                if x1 >= x2:
                    x1 = x2-1
                    x1 = max(0, x1)
                if y1 >= y2:
                    y1 = y2-1
                    y1 = max(0, y1)
                if x1 >= x2:
                    continue
                if y1 >= y2:
                    continue
                # print(x1, y1, x2, y2)
                feature_this_fake = feature1[mm, :, y1:y2, x1:x2]
                num = (y2-y1) * (x2-x1)
                feature_this_fake = feature_this_fake.sum(dim=2).sum(dim=1)
                feature_this_fake = feature_this_fake * 1.0 / num
                feature_this_real = feature3[mm, :, y1:y2, x1:x2]
                feature_this_real = feature_this_real.sum(dim=2).sum(dim=1)
                feature_this_real = feature_this_real * 1.0 / num
                label = int(label)
                cluster_class_real1[label] = cluster_class_real1[label] + feature_this_real
                cluster_class_fake1[label] = cluster_class_fake1[label] + feature_this_fake
                cluster_class_real_num[label] += 1
                cluster_class_fake_num[label] += 1

        for mm in range(self.num_classes):
            if cluster_class_real_num[mm] == 0:
                cluster_class_real.append(0)
                cluster_class_fake.append(0)
                continue
            logits_map_real_this = cluster_class_real1[mm] * (1.0 / cluster_class_real_num[mm])
            cluster_class_real.append(logits_map_real_this)

            logits_map_fake_this = cluster_class_fake1[mm] * (1.0 / cluster_class_fake_num[mm])
            cluster_class_fake.append(logits_map_fake_this)

        for mm in range(self.num_classes):
            if not (torch.is_tensor(cluster_class_real[mm])) or not (torch.is_tensor(cluster_class_fake[mm])):
                continue
            loss_class += torch.mean(torch.abs(cluster_class_real[mm] - cluster_class_fake[mm]))
            for nn in range(self.num_classes):
                if (mm == nn) or not (torch.is_tensor(cluster_class_real[nn])) or not (torch.is_tensor(cluster_class_fake[nn])):
                    continue
                distance_map2 = (5 - torch.mean(torch.abs(cluster_class_fake[mm] - cluster_class_fake[nn])))
                zeros_map = torch.zeros_like(distance_map2).cuda().float()
                loss_cluster2 += torch.max(distance_map2, zeros_map)
        loss_class = loss_class * 0.1
        loss_cluster2 = loss_cluster2 * 0.005
        # print('2', loss_class, loss_cluster2)
        loss_G_seg2 = loss_class + loss_cluster + loss_cluster2
        loss_G_seg2 = loss_G_seg2 * 10

        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_rec,
                                 loss_G_seg1, loss_G_seg2, loss_D_real, loss_D_fake), None if not infer else fake_image]

    def inference(self, label, inst, image=None):
        # Encode Inputs        
        image = Variable(image) if image is not None else None
        input_label, inst_map, real_image, _, adversarial_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(adversarial_image)
        else:
            fake_image = self.netG.forward(adversarial_image)
        return fake_image

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)
