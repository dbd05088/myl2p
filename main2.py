import logging.config
import os
import random
import pickle
from collections import defaultdict

import numpy as np
import torch
from randaugment import RandAugment
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from ml_collections import config_flags
from configs import config
from utils.augment import Cutout, select_autoaugment
from utils.data_loader import get_test_datalist, get_statistics
from utils.data_loader import get_train_datalist
from utils.method_manager import select_method

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

def main_worker(gpu, ngpus_per_node, train_transform, test_transform, criterion, n_classes, args):

    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    if args.distributed:
        '''
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        '''
        '''
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = gpu * ngpus_per_node + args.rank
            print("args.rank", args.rank)
        '''
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=ngpus_per_node, rank=gpu)

    if gpu is not None:
        device = torch.device(gpu)
        args.batchsize = int(args.batchsize / ngpus_per_node)
        args.n_worker = int((args.n_worker + ngpus_per_node - 1) / ngpus_per_node)
    else:
        device = torch.device("cpu")
    
    '''
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        if args.gpu_transform:
            args.gpu_transform = False
            logger.warning("Augmentation on GPU not available!")
    '''

    method = select_method(
        args, criterion, device, train_transform, test_transform, n_classes
    )

    logger.info(f"[2] Incrementally training {args.num_tasks} tasks")
    task_records = defaultdict(list)
    eval_results = defaultdict(list)
    samples_cnt = 0
    test_datalist = get_test_datalist(args.dataset)

    for cur_iter in range(args.num_tasks):
        if args.mode == "joint" and cur_iter > 0:
            return

        print("\n" + "#" * 50)
        print(f"# Task {cur_iter} iteration")
        print("#" * 50 + "\n")
        logger.info("[2-1] Prepare a datalist for the current task")

        # get datalist
        cur_train_datalist = get_train_datalist(args.dataset, args.num_tasks, args.m, args.n, args.seed, cur_iter)

        # Reduce datalist in Debug mode
        '''
        if args.debug:
            cur_train_datalist = cur_train_datalist[:2000]
            random.shuffle(test_datalist)
            test_datalist = test_datalist[:2000]
        '''

        method.online_before_task(cur_iter)
        for i, data in enumerate(cur_train_datalist):
            samples_cnt += 1
            method.online_step(data, samples_cnt, args.n_worker)
            if samples_cnt % args.eval_period == 0:
                eval_dict = method.online_evaluate(test_datalist, samples_cnt, 16, args.n_worker)
                eval_results["test_acc"].append(eval_dict['avg_acc'])
                eval_results["avg_acc"].append(eval_dict['cls_acc'])
                eval_results["data_cnt"].append(samples_cnt)
        method.online_after_task(cur_iter)
        eval_dict = method.online_evaluate(test_datalist, samples_cnt, 512, args.n_worker)
        task_acc = eval_dict['avg_acc']

        logger.info("[2-4] Update the information for the current task")
        task_records["task_acc"].append(task_acc)
        task_records["cls_acc"].append(eval_dict["cls_acc"])

        logger.info("[2-5] Report task result")
        writer.add_scalar("Metrics/TaskAcc", task_acc, cur_iter)

def main():

    args = config.base_parser()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6009'

    # for DDP!!
    # args.world_size = int(os.environ["WORLD_SIZE"])
    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node", ngpus_per_node)

    logging.config.fileConfig("./configs/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"results/{args.dataset}/{args.note}", exist_ok=True)
    os.makedirs(f"tensorboard/{args.dataset}/{args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{args.dataset}/{args.note}/seed_{args.seed}.log', mode="w")

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    writer = SummaryWriter(f'tensorboard/{args.dataset}/{args.note}/seed_{args.seed}')

    logger.info(args)

    #logger.info(f"Set the device ({device})")

    # Fix the random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Transform Definition
    mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset)
    inp_size = 224 # input size 고정
    train_transform = []
    if "cutout" in args.transforms:
        train_transform.append(Cutout(size=16))
        if args.gpu_transform:
            args.gpu_transform = False
            logger.warning("cutout not supported on GPU!")
    if "randaug" in args.transforms:
        train_transform.append(RandAugment())
        if args.gpu_transform:
            args.gpu_transform = False
            logger.warning("randaug not supported on GPU!")
    if "autoaug" in args.transforms:
        if 'cifar' in args.dataset:
            train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('cifar10')))
        elif 'imagenet' in args.dataset:
            train_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')))
    if args.gpu_transform:
        train_transform = transforms.Compose([
            transforms.RandomCrop(inp_size, padding=4),
            transforms.RandomHorizontalFlip(),
            *train_transform,
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((inp_size, inp_size)),
                transforms.RandomCrop(inp_size, padding=4),
                transforms.RandomHorizontalFlip(),
                *train_transform,
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    logger.info(f"Using train-transforms {train_transform}")

    test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    logger.info(f"[1] Select a CIL method ({args.mode})")
    criterion = nn.CrossEntropyLoss(reduction="mean")

    # original main worker 자리
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        print("world_size", args.world_size)

        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, train_transform, test_transform, criterion, n_classes, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, train_transform, test_transform, criterion, n_classes, args)

    np.save(f"results/{args.dataset}/{args.note}/seed_{args.seed}.npy", task_records["task_acc"])

    if args.mode == 'gdumb':
        eval_results, task_records = method.evaluate_all(test_datalist, args.memory_epoch, args.batchsize, args.n_worker)
    if args.eval_period is not None:
        np.save(f'results/{args.dataset}/{args.note}/seed_{args.seed}_eval.npy', eval_results['test_acc'])
        np.save(f'results/{args.dataset}/{args.note}/seed_{args.seed}_eval_time.npy', eval_results['data_cnt'])

    # Accuracy (A)
    A_auc = np.mean(eval_results["test_acc"])
    A_avg = np.mean(task_records["task_acc"])
    A_last = task_records["task_acc"][args.num_tasks - 1]

    # Forgetting (F)
    cls_acc = np.array(task_records["cls_acc"])
    acc_diff = []
    for j in range(n_classes):
        if np.max(cls_acc[:-1, j]) > 0:
            acc_diff.append(np.max(cls_acc[:-1, j]) - cls_acc[-1, j])
    F_last = np.mean(acc_diff)

    logger.info(f"======== Summary =======")
    logger.info(f"A_auc {A_auc} | A_avg {A_avg} | A_last {A_last} | F_last {F_last}")


if __name__ == "__main__":
    main()
