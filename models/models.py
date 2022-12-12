import torch

def create_model(opt):
    if opt.model == 'pix2pixHD_cityscapes':
        from .pix2pixHD_model_cityscapes import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHD_voc2012':
        from .pix2pixHD_model_voc2012 import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHD_cifar10':
        from .pix2pixHD_model_cifar10 import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHD_cifar100':
        from .pix2pixHD_model_cifar100 import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHD_tinyimagenet':
        from .pix2pixHD_model_tinyimagenet import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    elif opt.model == 'pix2pixHD_voc_det':
        from .pix2pixHD_model_voc_detection import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    else:
        model = None
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
