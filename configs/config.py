import argparse
import ml_collections

def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B_16",
        help="Select pretrained model",
    )
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--total_class", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="cifar100", help="dataset")
    parser.add_argument("--gaussian_schedule", type=bool, default=False)
    parser.add_argument("--gaussian_mode", type=str, default="")

    parser.add_argument("--offline_eval", type=bool, default=False)
    parser.add_argument("--recreate_eval", type=bool, default=False)
    parser.add_argument("--reinit_optimizer", type=bool, default=True)
    parser.add_argument("--eval_last_only", type=bool, default=False)
    parser.add_argument("--save_last_ckpt_only", type=bool, default=True)

    parser.add_argument("--lr", type=float, default=0.03, help="learning rate")
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--grad_clip_max_norm", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--learning_rate_schedule", type=str, default="constant")
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--weight_decay", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--num_eval_steps", type=int, default=-1)
    parser.add_argument("--eval_pad_last_batch", type=bool, default=False)
    parser.add_argument("--log_loss_every_steps", type=int, default=3)
    parser.add_argument("--checkpoint_every_steps", type=int, default=5000)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000) 

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trial", type=int, default=0)

    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--init_checkpoint", default=ml_collections.FieldReference(None, field_type=str)) #TODO

    # configuration for CL
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--num_classes_per_task", type=int, default=10)
    parser.add_argument("--eval_every_steps", type=int, default=-1)    
    parser.add_argument("--num_train_steps_per_task", type=int, default=-1)  
    parser.add_argument("--train_mask", type=bool, default=True)
    parser.add_argument("--eval_task_inc", type=bool, default=False)

    # if normalizing pre-logits
    parser.add_argument("--norm_pre_logits", type=bool, default=False)
    parser.add_argument("--weight_norm", type=bool, default=False)   
    parser.add_argument("--temperature", type=int, default=1) 
    # if using 0-1 normalization for input image
    parser.add_argument("--norm_01", type=bool, default=True)
    parser.add_argument("--reverse_task", type=bool, default=False) 

    # configuration for [cls] token
    parser.add_argument("--use_cls_token", type=bool, default=True)
    parser.add_argument("--task_specific_cls_token", type=bool, default=False)

    # classification option for ViT
    parser.add_argument("--vit_classifier", type=str, default="prompt")

    # do not use G-Prompt in L2P
    parser.add_argument("--use_g_prompt", type=bool, default=False)

    # use basic position and prompt-tuning of E-Prompt for L2P
    parser.add_argument("--use_e_prompt", type=bool, default=True) 
    parser.add_argument("--e_prompt_layer_idx", type=list, default=[0]) 
    parser.add_argument("--use_prefix_tune_for_e_prompt", type=bool, default=False)  

    # configuration for L2P
    parser.add_argument("--prompt_pool", type=bool, default=True)
    parser.add_argument("--pool_size", type=int, default=10)     
    parser.add_argument("--length", type=int, default=10)   
    parser.add_argument("--top_k", type=int, default=4)  
    parser.add_argument("--initializer", type=str, default="uniform")    
    parser.add_argument("--prompt_key", type=bool, default=True)
    parser.add_argument("--use_prompt_mask", type=bool, default=False)
    parser.add_argument("--mask_first_epoch", type=bool, default=False)  

    parser.add_argument("--shared_prompt_pool", type=bool, default=True)
    parser.add_argument("--shared_prompt_key", type=bool, default=False)
    parser.add_argument("--batchwise_prompt", type=bool, default=False)
    parser.add_argument("--prompt_key_init", type=str, default="uniform")
    parser.add_argument("--embedding_key", type=str, default="cls")
    parser.add_argument("--predefined_key_path", type=str, default="")

    # freeze model parts
    parser.add_argument("--freeze_part", type=list, default=["encoder", "embedding", "cls"])
    parser.add_argument("--freeze_bn_stats", type=bool, default=False)

    # subsample dataset or not
    parser.add_argument("--subsample_rate", type=int, default=-1)
     
    # key loss
    parser.add_argument("--pull_constraint", type=bool, default=True)
    parser.add_argument("--pull_constraint_coeff", type=float, default=1.0)

    # prompt utils
    parser.add_argument("--prompt_histogram", type=bool, default=True)
    parser.add_argument("--prompt_mask_mode", type=str, default=None)
    parser.add_argument("--save_prompts", type=bool, default=False)   

    # Mode and Exp. Settings.
    parser.add_argument(
        "--mode",
        type=str,
        default="er",
        help="Select CIL method",
    ) 
    parser.add_argument("--n", type=int, default=0, help="The percentage of disjoint split. Disjoint=100, Blurry=0")
    parser.add_argument("--m", type=int, default=10, help="The percentage of blurry samples in blurry split. Uniform split=100, Disjoint=0")
    parser.add_argument("--n_worker", type=int, default=2, help="The number of workers")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--data_dir", type=str, help="location of the dataset")
    # Dataset
    parser.add_argument(
        "--log_path",
        type=str,
        default="results",
        help="The path logs are saved.",
    )
    parser.add_argument("--gpu_transform", type =bool, default = False, help="perform data transform on gpu (for faster AutoAug).")

    parser.add_argument(
        "--use_amp", type = bool, default = False, help="Use automatic mixed precision."
    )

    # Note
    parser.add_argument("--note", type=str, help="Short description of the exp")
    # Transforms
    parser.add_argument(
        "--transforms",
        nargs="*",
        default=['cutmix', 'autoaug'],
        help="Additional train transforms [cutmix, cutout, randaug]",
    )
    # Train
    parser.add_argument("--dist_url", type=str, default="tcp://127.0.0.1:7009")
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")
    parser.add_argument("--stream_batchsize", type=int, default=32, help="batch size")
    parser.add_argument("--multiprocessing_distributed", type=bool, default=False)
    parser.add_argument("--distributed", type=bool, default=True)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.")
    parser.add_argument(
        "--memory_size", type=int, default=500, help="Episodic memory size"
    )
    parser.add_argument("--imp_update_period", type=int, default=1,
                        help="period between importance update, in units of model updates (increase for heavy datasets like ImageNet)")
    parser.add_argument("--eval_period", type=int, default=100, help="evaluation period for true online setup")
    parser.add_argument("--stream_eval_period", type=int, default=10000, help="evaluation period for true online setup")
    parser.add_argument(
        "--topk", type=int, default=1, help="set k when we want to set topk accuracy"
    )
    parser.add_argument("--temp_batchsize", type=int, help="temporary batch size, for true online")
    # RM & GDumb
    parser.add_argument("--memory_epoch", type=int, default=256, help="number of training epochs after task for Rainbow Memory")
    '''
    # Model
    parser.add_argument(
        "--model_name", type=str, default="resnet18", help="Model name"
    )



    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument(
        "--init_model",
        action="store_true",
        help="Initilize model parameters for every iterations",
    )
    parser.add_argument(
        "--init_opt",
        action="store_true",
        help="Initilize optimizer states for every iterations",
    )


    # Regularization
    parser.add_argument(
        "--reg_coef",
        type=int,
        default=100,
        help="weighting for the regularization loss term",
    )

    # Debug
    parser.add_argument("--debug", action="store_true", help="Turn on Debug mode")


    # Eval period





    # GDumb
    parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs, for GDumb eval')
    parser.add_argument('--workers_per_gpu', type=int, default=1, help='number of workers per GPU, for GDumb eval')

    # CLIB

    parser.add_argument('--lr_step', type=float, default=0.95, help='step of iterating lr for adaptive LR')
    parser.add_argument('--lr_length', type=int, default=10, help='period of iterating lr for adaptive LR')
    parser.add_argument('--lr_period', type=int, default=10, help='period of iterating lr for adaptive LR')

    # BiC
    parser.add_argument("--distilling", type=bool, default=True, help="use distillation for BiC.")

    # AGEM
    parser.add_argument('--agem_batch', type=int, default=240, help='A-GEM batch size for calculating gradient')

    # MIR
    parser.add_argument('--mir_cands', type=int, default=50, help='# candidates to use for MIR')
    '''
    args = parser.parse_args()
    return args
