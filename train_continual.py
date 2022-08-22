import ml_collections
import torch_optimizer
from torch import optim

# global variable for maintaining summary steps
summary_step = 0


def create_optimizer(weight_decay, freezing, opt_name, lr, params, sgd_momentum = None, freeze_part = None):
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
            opt = optim.Adam(params, lr=lr, weight_decay=weight_decay)
            
        elif opt_name == "radam":
            #opt = torch_optimizer.RAdam(params, lr=lr, weight_decay=0.00001)
            opt = torch_optimizer.RAdam(params, lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            #opt = optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
            opt = optim.SGD(params, lr=lr, momentum=sgd_momentum, nesterov=True, weight_decay=1e-4)

        else:
            raise NotImplementedError("Please select the opt_name [adam, sgd]")
    else:
        freeze_part = freeze_part
        freeze_params = []
        normal_params = []
        for name, param in params:
            freeze_flag = False
            for freeze in freeze_part:
                if freeze in name:
                    freeze_flag = True
                    break
            if freeze_flag:
                freeze_params.append(param)
            else:
                normal_params.append(param)

        freeze_params = [param for name, param in params if name not in freeze_part]
        normal_params = [param for name, param in params if name in freeze_part]
        if opt_name == "adam":            
            opt = torch_optimizer.Adam(normal_params, lr=lr, weight_decay=weight_decay)
            opt.add_param_group({'params': freeze_params, 'weight_decay':0})
        elif opt_name == "sgd":
            opt = optim.SGD(normal_params, lr=lr, momentum=sgd_momentum, nesterov=True, weight_decay=1e-4)
            opt.add_param_group({'params': freeze_params, 'momentum':0, 'nesterov':False})
    return opt
