import logging

from methods.l2p import l2p
#from methods.bic import BiasCorrection
#from methods.er_baseline import ER
#from methods.rainbow_memory import RM
#from methods.ewc import EWCpp
#from methods.mir import MIR
#from methods.clib import CLIB

logger = logging.getLogger()

def select_method(args, criterion, device, train_transform, test_transform, n_classes):
    kwargs = vars(args)
    method = l2p(
        criterion=criterion,
        device=device,
        train_transform=train_transform,
        test_transform=test_transform,
        n_classes=n_classes,
        **kwargs,
    )
    return method
