# When we make a new one, we should inherit the Finetune class.
import logging
import copy
import time
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import functools
from utils.data_loader import ImageDataset, StreamDataset, MemoryDataset, cutmix_data, get_statistics
from utils.train_utils import select_model2, create_optimizer, select_scheduler
from pytorch_pretrained_vit import ViT

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class l2p:
    def __init__(
            self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        self.num_learned_class = 0
        self.num_learning_class = 1
        self.n_classes = n_classes
        self.exposed_classes = []
        self.seen = 0
        self.topk = kwargs["topk"]

        self.device = device
        self.dataset = kwargs["dataset"]
        self.model_name = kwargs["model_name"]
        self.sched_name = kwargs["sched_name"]
        if self.sched_name == "default":
            self.sched_name = 'exp_reset'
        self.lr = kwargs["lr"]

        self.train_transform = train_transform
        self.cutmix = "cutmix" in kwargs["transforms"]
        self.test_transform = test_transform

        self.memory_size = kwargs["memory_size"]
        self.data_dir = kwargs["data_dir"]

        self.online_iter = kwargs["online_iter"]
        self.batch_size = kwargs["batchsize"]
        self.temp_batchsize = kwargs["temp_batchsize"]
        if self.temp_batchsize is None:
            self.temp_batchsize = self.batch_size//2
        if self.temp_batchsize > self.batch_size:
            self.temp_batchsize = self.batch_size
        self.memory_size -= self.temp_batchsize

        self.gpu_transform = kwargs["gpu_transform"]
        self.use_amp = kwargs["use_amp"]

        #l2p 추가
        self.freeze_part = kwargs["freeze_part"]
        if self.freeze_part != []:
            self.freezing = True
        else:
            self.freezing = False
        self.sgd_momentum = kwargs["sgd_momentum"]
        self.optim = kwargs["optim"]
        self.weight_decay = kwargs["weight_decay"]
        self.norm_pre_logits = kwargs["norm_pre_logits"]
        self.temperature = kwargs["temperature"]
        self.use_e_prompt = kwargs["use_e_prompt"]
        self.e_prompt_layer_idx = kwargs["e_prompt_layer_idx"]
        self.use_prefix_tune_for_e_prompt = kwargs["use_prefix_tune_for_e_prompt"]
        self.use_cls_token = kwargs["use_cls_token"]
        self.vit_classifier = kwargs["vit_classifier"]
        self.num_tasks = kwargs["num_tasks"]
        self.num_classes_per_task = kwargs["num_classes_per_task"]
        self.num_total_class = kwargs["total_class"]

        self.prompt_pool = kwargs["prompt_pool"]
        self.prompt_pool_param_pool_size = kwargs["pool_size"]
        self.prompt_pool_param_length = kwargs["length"]
        self.prompt_pool_param_top_k = kwargs["top_k"]
        self.prompt_pool_param_initializer = kwargs["initializer"]
        self.prompt_pool_param_prompt_key = kwargs["prompt_key"]
        self.prompt_pool_param_use_prompt_mask = kwargs["use_prompt_mask"]
        self.prompt_pool_param_mask_first_epoch = kwargs["mask_first_epoch"]
        self.prompt_pool_param_shared_prompt_pool = kwargs["shared_prompt_pool"]
        self.prompt_pool_param_shared_prompt_key = kwargs["shared_prompt_key"]
        self.prompt_pool_param_batchwise_prompt = kwargs["batchwise_prompt"]
        self.prompt_pool_param_prompt_key_init = kwargs["prompt_key_init"]
        self.prompt_pool_param_embedding_key = kwargs["embedding_key"]
        param_dic = {"pool_size": self.prompt_pool_param_pool_size,
                    "length": self.prompt_pool_param_length,
                    "top_k": self.prompt_pool_param_top_k,
                    "initializer": self.prompt_pool_param_initializer,
                    "prompt_key": self.prompt_pool_param_prompt_key,
                    "use_prompt_mask": self.prompt_pool_param_use_prompt_mask,
                    "mask_first_epoch": self.prompt_pool_param_mask_first_epoch,
                    "shared_prompt_pool": self.prompt_pool_param_shared_prompt_pool,
                    "shared_prompt_key": self.prompt_pool_param_shared_prompt_key,
                    "batchwise_prompt": self.prompt_pool_param_batchwise_prompt,
                    "prompt_key_init": self.prompt_pool_param_prompt_key_init,
                    "embedding_key": self.prompt_pool_param_embedding_key
                    }
        self.prompt_pool_param = param_dic # kwargs["prompt_pool_param"]에 포함되는 모든 애들

        self.pretrained_model = ViT('B_16', pretrained=True)
        self.pretrained_dict = self.pretrained_model.state_dict()

        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        self.model =  select_model2(self.model_name, 
                                    self.n_classes, 
                                    self.sgd_momentum,
                                    self.optim,
                                    self.weight_decay,
                                    self.norm_pre_logits,
                                    self.temperature,
                                    self.use_e_prompt,
                                    self.e_prompt_layer_idx,
                                    self.use_prefix_tune_for_e_prompt,
                                    self.use_cls_token,
                                    self.vit_classifier,
                                    self.num_tasks,
                                    self.num_classes_per_task,
                                    self.num_total_class,
                                    self.device,
                                    self.prompt_pool_param)
        #print("self.model")
        #print(self.model)
        self.model = self.model.to(self.device)
        # weight_decay, freezing, opt_name, lr, params, sgd_momentum = None, freeze_part = None
        # 여기서의 param은 model parameter이다.
        '''
        print("model params")
        for name, param in self.model.named_parameters():
            print(name)
        '''
        self.optimizer = create_optimizer(self.weight_decay, self.freezing, self.optim, self.lr, self.model, self.sgd_momentum, self.freeze_part)
        if 'imagenet' in self.dataset:
            self.lr_gamma = 0.99995
        else:
            self.lr_gamma = 0.9999
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

        self.criterion = criterion.to(self.device)
        self.memory = MemoryDataset(self.dataset, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform)
        self.temp_batch = []
        self.num_updates = 0
        self.train_count = 0
        self.batch_size = kwargs["batchsize"]

        self.start_time = time.time()
        num_samples = {'cifar10': 50000, 'cifar100': 50000, 'tinyimagenet': 100000, 'imagenet': 1281167}
        self.total_samples = num_samples[self.dataset]

    #def get_train_eval_components(self):

    def online_step(self, sample, sample_num, n_worker):
        
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
        
        self.temp_batch.append(sample)

        if len(self.temp_batch) == self.batch_size: # batch size만큼 대기
            train_loss, train_acc = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=1, stream_batch_size=self.batch_size)
            self.report_training(sample_num, train_loss, train_acc)
            for stored_sample in self.temp_batch:
                self.update_memory(stored_sample)
            self.temp_batch = []

    '''
    def online_step(self, sample, sample_num, n_worker):
        
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])
        
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.temp_batchsize:
            train_loss, train_acc = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=self.temp_batchsize)
            self.report_training(sample_num, train_loss, train_acc)
            for stored_sample in self.temp_batch:
                self.update_memory(stored_sample)
            self.temp_batch = []
            self.num_updates -= int(self.num_updates)
    '''

    def load_pretrain_state_dict(self):

        model_dict = self.model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in self.pretrained_dict.items() if k in model_dict.keys() and 'fc' not in k and 'norm' not in k} #TODO norm weight 부분은 
        remain_dict = {k: v for k, v in self.pretrained_dict.items() if k not in model_dict.keys()} # for print checking
        #print("pretrained_keys", self.pretrained_model.state_dict().keys())
        #print("model_keys", self.model.state_dict().keys())


        #print("remain_dict")
        #print(remain_dict.keys())
        #print("pretrained_dict")
        #print(pretrained_dict.keys())

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        '''
        # 3. load the new state dict
        self.model.load_state_dict(model_dict)
        '''
        # pretrained_model의 dict
        return model_dict

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        '''
        prev_weight = copy.deepcopy(self.model.fc.weight.data)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)

        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
        for param in self.optimizer.param_groups[1]['params']:
            if param in self.optimizer.state.keys():
                del self.optimizer.state[param]
        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': self.model.fc.parameters()})
        '''
        self.memory.add_new_class(cls_list=self.exposed_classes)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=self.gpu_transform)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

        for i in range(iterations):
            self.model.train()
            x = []
            y = []
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x.append(stream_data['image'])
                y.append(stream_data['label'])
            if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                x.append(memory_data['image'])
                y.append(memory_data['label'])
            x = torch.cat(x)
            y = torch.cat(y)
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            logit, loss = self.model_forward(x,y)

            _, preds = logit.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.update_schedule()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        return total_loss / iterations, correct / num_data

    def model_forward(self, x, y):
        pretrained_model_dict = self.load_pretrain_state_dict()
        original_model_dict = self.model.state_dict()

        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    res = self.model(x)
                    logit = res['logits']
                    loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
            else:
                res = self.model(x)
                logit = res['logits']
                loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(logit, labels_b)
        else:
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    res = self.model(x)
                    logit = res['logits']
                    loss = self.criterion(logit, y)
            else:
                # step 1 - pretrained pre-logit
                self.model.load_state_dict(pretrained_model_dict)
                self.model.eval()
                res_pre = self.model(x)
                cls_features = res_pre['pre_logits']

                self.model.train()
                res = self.model(x, cls_features=cls_features)
                logit = res['logits']

            # here is the trick to mask out classes of non-current tasks
            #print("self.exposed_class", self.exposed_classes)
            not_mask = torch.Tensor(np.setdiff1d(np.arange(self.num_total_class), np.arange(len(self.exposed_classes))))
            logit[:, len(self.exposed_classes):] = -np.inf

            #logit = logit.masked_fill(not_mask==False, -np.inf)
            #print("after logit", logit)
            loss = self.criterion(logit, y)

        return logit, loss

    def report_training(self, sample_num, train_loss, train_acc):
        writer.add_scalar(f"train/loss", train_loss, sample_num)
        writer.add_scalar(f"train/acc", train_acc, sample_num)
        logger.info(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc):
        writer.add_scalar(f"test/loss", avg_loss, sample_num)
        writer.add_scalar(f"test/acc", avg_acc, sample_num)
        logger.info(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | "
        )

    def update_memory(self, sample):
        self.reservoir_memory(sample)

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()

    def online_evaluate(self, test_list, sample_num, batch_size, n_worker):
        test_df = pd.DataFrame(test_list)
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        test_dataset = ImageDataset(
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        eval_dict = self.evaluation(test_loader, self.criterion)
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"])
        return eval_dict

    def online_before_task(self, cur_iter):
        # Task-Free
        self.reset_opt() # optimizer reset

        sdict = copy.deepcopy(self.optimizer.state_dict())
        '''
        print("sdict")
        print(sdict) # freezing과 non_freezing 

        print("prompt_pool_module.prompt shape")
        print(self.model.state_dict()['prompt_pool_module.prompt'].shape)
        print("prompt_pool_module.prompt key shape")
        print(self.model.state_dict()['prompt_pool_module.prompt_key'].shape)        
        '''
        # Transfer previous learned prompt params to the new prompt 
        if self.prompt_pool and self.prompt_pool_param['shared_prompt_pool']:
            if cur_iter > 0:
                prev_start = (cur_iter- 1) * self.prompt_pool_param['top_k']
                prev_end = cur_iter * self.prompt_pool_param['top_k']
                cur_start = prev_end
                cur_end = (cur_iter + 1) * self.prompt_pool_param['top_k']

                if (prev_end > self.prompt_pool_param['pool_size']) or (
                    cur_end > self.prompt_pool_param['pool_size']):
                    pass
                else:
                    if self.use_prefix_tune_for_e_prompt:
                        self.model.state_dict()['prompt_pool_module.prompt'][:, :, cur_start:cur_end] = self.model.state_dict()['prompt_pool_module.prompt'][:, :, prev_start:prev_end]
                    else:
                        self.model.state_dict()['prompt_pool_module.prompt'][:, cur_start:cur_end] = self.model.state_dict()['prompt_pool_module.prompt'][:, prev_start:prev_end]

        # Transfer previous learned prompt param keys to the new prompt
        if self.prompt_pool and self.prompt_pool_param_prompt_key and self.prompt_pool_param_shared_prompt_key:
            if cur_iter > 0:
                prev_start = (cur_iter- 1) * self.prompt_pool_param_top_k
                prev_end = cur_iter * self.prompt_pool_param_top_k
                cur_start = prev_end
                cur_end = (cur_iter + 1) * self.prompt_pool_param_top_k
                self.model.state_dict()['prompt_pool_module.prompt_key'][cur_start:cur_end] = self.model.state_dict()['prompt_pool_module.prompt_key'][prev_start:prev_end]


    def online_after_task(self, cur_iter):
        # Task-Free
        pass

    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
        else:
            self.memory.replace_sample(sample)

    def reset_opt(self):
        self.optimizer = create_optimizer(self.weight_decay, self.freezing, self.optim, self.lr, self.model, self.sgd_momentum, self.freeze_part)#create_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)

    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                res = self.model(x)
                logit = res["logits"]

                loss = criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret

    def _interpret_pred(self, y, pred):
        # xlable is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects
