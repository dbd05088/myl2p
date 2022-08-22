from numpy import isin
import torch_optimizer
from easydict import EasyDict as edict
from torch import optim
from models import vit
import torch
#from models import mnist, cifar, imagenet
import functools
def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i

def create_optimizer(weight_decay, freezing, opt_name, lr, model, sgd_momentum = None, freeze_part = None):
    """Optionally creates the optimizer to use for every task.

    Args:
        config: Configuration to use.
        params: Parameters associated with the optimizer.

    Returns:
        The newly created optimizer.
    """
    #print("optim_wd_ignore!!", config.get("optim_wd_ignore"))
    if (not freezing):
        if opt_name == "adam":
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
        elif opt_name == "radam":
            #opt = torch_optimizer.RAdam(params, lr=lr, weight_decay=0.00001)
            opt = torch_optimizer.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            #opt = optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
            opt = optim.SGD(model.parameters(), lr=lr, momentum=sgd_momentum, nesterov=True, weight_decay=1e-4)

        else:
            raise NotImplementedError("Please select the opt_name [adam, sgd]")
    else:
        freeze_part = freeze_part
        freeze_params = []
        normal_params = []
        freeze_names = []
        normal_names = []
        for name, param in model.named_parameters():
            freeze_flag = False
            for freeze in freeze_part:
                if freeze in name:
                    freeze_flag = True
                    break
            if freeze_flag:
                freeze_names.append(name)
                freeze_params.append(param)
            else:
                normal_names.append(name)
                normal_params.append(param)

        #freeze_params = [param for name, param in model.named_parameters() if name not in freeze_part]
        #normal_params = [param for name, param in model.named_parameters() if name in freeze_part]
        
        '''
        print("freeze part")
        print(freeze_part)
        print("freeze_params")
        print(freeze_names)
        '''
        #print("normal_parms")
        #print(normal_names)
        
        if opt_name == "adam":
            '''
            # 순서 변경
            opt = optim.Adam(normal_params, lr=lr, weight_decay=weight_decay)
            opt.add_param_group({'params': freeze_params, 'weight_decay':0})
            '''
            #opt = optim.Adam(freeze_params, lr=0, weight_decay=0)
            #opt.add_param_group({'params': normal_params, 'lr':0.03, 'weight_decay':weight_decay})

            opt = optim.Adam(normal_params, lr=0.03, weight_decay=weight_decay)
            opt.add_param_group({'params': freeze_params, 'lr':0, 'weight_decay':weight_decay})           
        elif opt_name == "sgd":
            opt = optim.SGD(normal_params, lr=lr, momentum=sgd_momentum, nesterov=True, weight_decay=1e-4)
            opt.add_param_group({'params': freeze_params, 'lr':0, 'momentum':0, 'nesterov':False})
    return opt


def select_scheduler(sched_name, opt, hparam=None):
    if "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=hparam)
    elif sched_name == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=1, T_mult=2
        )
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            opt, milestones=[30, 60, 80, 90], gamma=0.1
        )
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler

def select_model2(
    model_name, 
    num_classes, 
    sgd_momentum, 
    optim, 
    weight_decay, 
    norm_pre_logits, 
    temperature, 
    use_e_prompt, 
    e_prompt_layer_idx, 
    use_prefix_tune_for_e_prompt, 
    use_cls_token,
    vit_classifier,
    num_tasks,
    num_classes_per_task,
    num_total_class,
    device,
    prompt_pool_param):
    """Creates and initializes the model.
    """
    # Create model function.
    if "resnet" in model_name:
        #model_cls = resnet_v1.create_model(model_name, )
        model = vit.create_model(model_name, sgd_momentum, optim, weight_decay, norm_pre_logits, temperature, use_e_prompt, e_prompt_layer_idx, use_prefix_tune_for_e_prompt, use_cls_token, vit_classifier, num_tasks, num_classes_per_task, num_total_class, device, prompt_pool_param)
    elif "ViT" in model_name:
        model = vit.create_model(model_name, sgd_momentum, optim, weight_decay, norm_pre_logits, temperature, use_e_prompt, e_prompt_layer_idx, use_prefix_tune_for_e_prompt, use_cls_token, vit_classifier, num_tasks, num_classes_per_task, num_total_class, device, prompt_pool_param)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    # function tool의 partial: 하나 이상의 인수가 이미 채워진 함수의 새 버전을 만들기 위해서 사용됨
    #model = functools.partial(model_cls, num_classes=num_classes)
    #return torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    '''
    print("model_summary")
    for key in model.state_dict().keys():
        print(key)
    '''
    return model

def select_model(model_name, dataset, num_classes=None):
    opt = edict(
        {
            "depth": 18,
            "num_classes": num_classes,
            "in_channels": 3,
            "bn": True,
            "normtype": "BatchNorm",
            "activetype": "ReLU",
            "pooltype": "MaxPool2d",
            "preact": False,
            "affine_bn": True,
            "bn_eps": 1e-6,
            "compression": 0.5,
        }
    )

    if "mnist" in dataset:
        model_class = getattr(mnist, "MLP")
    elif "cifar" in dataset:
        model_class = getattr(cifar, "ResNet")
    elif "imagenet" in dataset:
        model_class = getattr(imagenet, "ResNet")
    else:
        raise NotImplementedError(
            "Please select the appropriate datasets (mnist, cifar10, cifar100, imagenet)"
        )
    if model_name == "resnet18":
        opt["depth"] = 18
    elif model_name == "resnet32":
        opt["depth"] = 32
    elif model_name == "resnet34":
        opt["depth"] = 34
    elif model_name == "mlp400":
        opt["width"] = 400
    else:
        raise NotImplementedError(
            "Please choose the model name in [resnet18, resnet32, resnet34]"
        )

    model = model_class(opt)

    return model
