import numpy as np
import torch
import os
from torch.autograd import Variable
import torch.optim as optim
from util.image_pool import ImagePool
from .base_model import BaseModel

from . import networks
import torch.nn as nn
import random
import torch.nn.functional as F
import random

from .cls.wideresnet import WideResNet2
from .cls.resnet import ResNet50_class200


def PGD_kl(model, x_natural, y, step_size=0.0175, epsilon=0.031, perturb_steps=4):
    # define KL-loss
    # model.eval()

    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.zero_grad()
    return x_adv.detach()

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
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)

        ################################################################################################
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        self.num_cls = 200

        resume_path = 'model_cls_NoD/tinyimagenet/wideresnet/model.pt'
        self.cls_model = ResNet50_class200().cuda()
        self.cls_model.load_state_dict(torch.load(resume_path))
        self.cls_model.eval()
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

    def obtain_action_label_feature(self, action_label, h, w):
        batch_size = action_label.size(0)
        action_label_clone = action_label.view(batch_size, )
        action_label_feature = torch.zeros(batch_size, self.num_cls).cuda().float()
        index = torch.LongTensor(range(action_label_feature.size(0))).cuda()
        action_label_feature[index, action_label_clone.long()] = 1.0

        action_label_feature = action_label_feature.unsqueeze(-1).unsqueeze(-1)
        action_label_feature = action_label_feature.repeat(1, 1, h, w)
        return action_label_feature

    def encode_input3(self, image, label, infer=False):
        target_label = Variable(label.clone().data.cuda())
        real_image = Variable(image.data.cuda())
        real_image_clone = real_image.clone()

        ### the range of input is [0, 1]
        self.input_h = real_image.shape[2]
        self.input_w = real_image.shape[3]
        real_image = PGD_kl(self.cls_model, real_image, label)

        real_image[:, 0, :, :] = (real_image[:, 0, :, :] - self.mean[0]) / self.std[0]
        real_image[:, 1, :, :] = (real_image[:, 1, :, :] - self.mean[1]) / self.std[1]
        real_image[:, 2, :, :] = (real_image[:, 2, :, :] - self.mean[2]) / self.std[2]
        real_image = F.interpolate(real_image, size=(64, 64), mode='bilinear', align_corners=True)

        real_image_clone[:, 0, :, :] = (real_image_clone[:, 0, :, :] - self.mean[0]) / self.std[0]
        real_image_clone[:, 1, :, :] = (real_image_clone[:, 1, :, :] - self.mean[1]) / self.std[1]
        real_image_clone[:, 2, :, :] = (real_image_clone[:, 2, :, :] - self.mean[2]) / self.std[2]
        real_image_clone = F.interpolate(real_image_clone, size=(64, 64), mode='bilinear', align_corners=True)

        input_label = self.obtain_action_label_feature(label, self.input_h, self.input_w)
        return real_image_clone, real_image, target_label, input_label

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, data, label, infer=False):
        # Encode Inputs
        real_image, adversarial_image, target_label, input_label = self.encode_input3(data, label)

        fake_image = self.netG.forward(adversarial_image)
        fake_image2 = self.netG.forward(real_image)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate(input_label, fake_image2)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))        
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
        fake_image = fake_image.mul(std_map).add(mean_map)
        # fake_image [0, 1]

        result = self.cls_model(fake_image)
        loss_G_seg1 = nn.CrossEntropyLoss()(result, target_label) * 2

        fake_image2 = F.interpolate(fake_image2, size=(self.input_h, self.input_w), mode='bilinear', align_corners=True)
        fake_image2 = fake_image2.mul(std_map).add(mean_map)
        result2 = self.cls_model(fake_image2)
        loss_G_seg1 += nn.CrossEntropyLoss()(result2, target_label) * 2

        real_image = F.interpolate(real_image, size=(self.input_h, self.input_w), mode='bilinear', align_corners=True)
        real_image = real_image.mul(std_map).add(mean_map)

        result_real = self.cls_model(real_image)
        loss_G_seg1 += torch.mean(torch.abs(result - result_real)) * 4
        loss_G_seg1 += torch.mean(torch.abs(result2 - result_real)) * 4


        logits_map_real = result_real
        logits_map_fake = result
        loss_cluster = 0
        loss_cluster2 = 0
        loss_class = 0
        cluster_class_real = []
        cluster_class_fake = []
        for mm in range(self.num_cls):
            cls_this = (target_label == mm)
            cls_num = torch.sum(cls_this).item()
            if cls_num == 0:
                cluster_class_real.append(0)
                cluster_class_fake.append(0)
                continue

            logits_map_real_this = logits_map_real[cls_this]
            logits_map_real_this_mean = torch.mean(logits_map_real_this, dim=0).unsqueeze(0)
            cluster_class_real.append(logits_map_real_this_mean)

            logits_map_fake_this = logits_map_fake[cls_this]
            logits_map_fake_this_mean = torch.mean(logits_map_fake_this, dim=0).unsqueeze(0)
            cluster_class_fake.append(logits_map_fake_this_mean)
            logits_map_fake_this_mean_this = logits_map_fake_this_mean.expand_as(logits_map_fake_this)
            logits_map_fake_loss = torch.mean(torch.abs(logits_map_fake_this - logits_map_fake_this_mean_this))
            loss_cluster += logits_map_fake_loss

        for mm in range(self.num_cls):
            if not (torch.is_tensor(cluster_class_real[mm])) or not (torch.is_tensor(cluster_class_fake[mm])):
                continue
            loss_class += torch.mean(torch.abs(cluster_class_real[mm] - cluster_class_fake[mm]))
            for kk in range(self.opt.label_nc):
                if (mm == kk) or not (torch.is_tensor(cluster_class_real[kk])) or not (torch.is_tensor(cluster_class_fake[kk])):
                    continue
                distance_map2 = (5 - torch.mean(torch.abs(cluster_class_fake[mm] - cluster_class_fake[kk])))
                zeros_map = torch.zeros_like(distance_map2).cuda().float()
                loss_cluster2 += torch.max(distance_map2, zeros_map)
        loss_class = loss_class * 0.1
        loss_cluster2 = loss_cluster2 * 0.005
        loss_G_seg2 = loss_class + loss_cluster + loss_cluster2


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

        
