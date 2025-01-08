import argparse
import datetime
import math
import os
import random
import sys
import time
from collections import defaultdict
from copy import deepcopy
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import logging as logging_module  # Renamed to avoid conflict with misc.logger
from data import build_hf_tiny_imagenet_loader, build_loader
from misc.attentive_nas_eval import validate as nas_validate
from misc.config import get_config
from misc.lr_scheduler import build_scheduler
from misc.loss_ops import AdaptiveLossSoft
from misc.optimizer import build_optimizer
from misc.utils import (
    load_checkpoint,
    save_checkpoint,
    get_grad_norm,
    auto_resume_helper,
    reduce_tensor,
    load_teacher_checkpoint,
)
import misc.logger as logging
import models
import timm
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter, ModelEma

# Suppress warnings and set logging levels
logging_module.getLogger("PIL").setLevel(logging_module.WARNING)
warnings.filterwarnings("ignore")
logger = logging.get_logger(__name__)


def parse_option():
    parser = argparse.ArgumentParser(
        description="Super-HQViT Training and Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="/home/ali/Pictures/HQViT_NAS_Tiny_imgnt/configs/cfg.yaml",
        metavar="FILE",
        help="Path to config file",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default="/home/ali/Pictures/test",
        # required=True,
        help="Root directory for models and logs",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Enable distributed training",
    )
    parser.add_argument("--batch-size", type=int, help="Batch size for single GPU")
    parser.add_argument("--resume", type=str, help="Path to resume checkpoint")
    parser.add_argument(
        "--fp_teacher_dir",
        default="/home/ali/Downloads/ckpt_360(1).pth",
        type=str,
        help="Path to full-precision teacher weights",
    )
    parser.add_argument(
        "--accumulation-steps", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="Use gradient checkpointing to save memory",
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--print-freq", type=int, default=100, help="Print frequency")
    args, _ = parser.parse_known_args()

    # Setup the work directory
    args.output = args.working_dir
    writer = SummaryWriter(os.path.join(args.output, "log"))

    # Load configuration
    config = get_config(args)
    return args, config


def setup_worker_env(rank, ngpus_per_node, config):
    """
    Sets up the distributed training environment.
    """
    if config.distributed:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=config.WORLD_SIZE,
            rank=rank,
        )
        torch.cuda.set_device(rank)
    else:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            world_size=config.WORLD_SIZE,
            rank=rank,
        )
        # Single GPU or CPU training
        torch.cuda.set_device(rank)

    # Set seeds for reproducibility
    seed = config.SEED + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # Linear scaling of the learning rate based on batch size and world size
    config.defrost()
    linear_scaled_lr = (
        config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * config.WORLD_SIZE / 512.0
    )
    linear_scaled_warmup_lr = (
        config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * config.WORLD_SIZE / 512.0
    )
    linear_scaled_min_lr = (
        config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * config.WORLD_SIZE / 512.0
    )

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr *= config.TRAIN.ACCUMULATION_STEPS

    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # Setup logging
    logging.setup_logging(
        os.path.join(config.OUTPUT, "stdout.log"),
        mode="w",
    )

    # Backup the config
    if rank == 0:
        config_path = os.path.join(config.OUTPUT, "config.json")
        with open(config_path, "w") as f:
            f.write(config.dump())
        logging.get_logger(__name__).info(f"Full config saved to {config_path}")

    # Log the config
    logging.get_logger(__name__).info(config.dump())


def main_worker(rank, ngpus_per_node, config, args):
    """
    Main worker function for each process in distributed training.
    """
    setup_worker_env(rank, ngpus_per_node, config)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Build data loaders
    (
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
        mixup_fn,
    ) = build_loader(config)
    logging.get_logger(__name__).info(
        f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}"
    )

    # Create model
    model = models.model_factory.create_model(
        config, parent="model", full_precision=False
    )
    model.to(device)

    logging.get_logger(__name__).info(str(model))

    # Model EMA
    model_ema = ModelEma(model, decay=0.99985, device=device, resume="")

    # Build optimizer and scheduler
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # Wrap model with DDP

    model = DDP(
        model, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=True
    )
    model_without_ddp = model.module

    if config.MODEL.TEACHER_RESUME:
        teacher_model = models.model_factory.create_model(config, full_precision=True)
        teacher_model.to(device)
        # teacher_max_accuracy=load_teacher_checkpoint(config,teacher_model,logger)

    # Log model parameters and FLOPs
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.get_logger(__name__).info(f"Number of parameters: {n_parameters}")
    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logging.get_logger(__name__).info(f"Number of GFLOPs: {flops / 1e9}")

    # Define loss criterion
    if config.AUG.MIXUP > 0.0:
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss()

    # Load checkpoint if resuming
    max_accuracy = 0.0
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config,
            model_without_ddp,
            optimizer,
            lr_scheduler,
            logging.get_logger(__name__),
        )
        model_ema.ema = deepcopy(model_without_ddp)
        if config.EVAL_MODE:
            validate(config, data_loader_train, data_loader_val, model, device)
            return

    logging.get_logger(__name__).info("Start training")
    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(
            config=config,
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            mixup_fn=mixup_fn,
            lr_scheduler=lr_scheduler,
            model_ema=model_ema,
            device=device,
            args=args,
            teacher=teacher_model,
        )

        # Save checkpoint
        if (epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == config.TRAIN.EPOCHS:
            if rank == 0:
                save_checkpoint(
                    config,
                    epoch,
                    model_without_ddp,
                    max_accuracy,
                    optimizer,
                    lr_scheduler,
                    logging.get_logger(__name__),
                    model_ema=model_ema,
                )

        # Validation
        if (epoch + 1) % config.VAL_FREQ == 0 or (epoch + 1) == config.TRAIN.EPOCHS:
            validate(config, data_loader_train, data_loader_val, model, device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.get_logger(__name__).info(f"Training time {total_time_str}")
    if config.distributed:
        dist.destroy_process_group()


def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
    model_ema,
    device,
    args,
    teacher=None,
):
    """
    Trains the model for one epoch.
    """
    model.train()
    num_steps = len(data_loader)
    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    super_loss_meter = AverageMeter()
    grad_norm_meter = AverageMeter()
    teacher_criteration = AdaptiveLossSoft(alpha_min=-1.0, alpha_max=1.0)
    gamma = 1.0
    beta = 1.0

    start = time.time()

    optimizer.zero_grad()

    for idx, (samples, targets) in enumerate(data_loader):
        data_time.update(time.time() - start)
        samples = samples.to(device)
        targets = targets.to(device)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        cfg = model.module.sample_max_subnet()
        output, _, _ = model(samples)
        loss = criterion(output, targets)
        if teacher is not None:
            with torch.no_grad():
                teacher_cfg = teacher.sample_max_subnet()
                teacher_output, _, _ = teacher(samples)
                teacher_soft_logit = teacher_output.clone().detach()

            super_teacher_loss = teacher_criteration(output, teacher_soft_logit)
            loss = loss + gamma * super_teacher_loss
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # Backward pass
        loss.backward()

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.TRAIN.CLIP_GRAD
            )
            grad_norm_meter.update(grad_norm.item())

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # Update learning rate scheduler
            lr_scheduler.step_update(epoch * num_steps + idx)

            # Update EMA
            if model_ema is not None:
                model_ema.update(model)

        

        super_sandwich_rule = getattr(config, "super_sandwich_rule")
        num_subnet_training = max(2, getattr(config, "num_arch_training", 2))
        model.module.set_dropout_rate(0.0, 0.0, True)

        if idx % config.TRAIN.SUBNET_REPEAT_SAMPLE == 0:
            cfgs = []

        for arch_id in range(1, num_subnet_training):
            if arch_id == 1 and super_sandwich_rule:
                model.module.sample_min_subnet()
                if teacher:
                    teacher.sample_min_subnet()
            else:
                if idx % config.TRAIN.SUBNET_REPEAT_SAMPLE == 0:
                    cfg = model.module.sample_active_subnet()
                    cfgs.append(cfg)
                else:
                    cfg = cfgs[arch_id - 1]
                    model.module.set_active_subnet(
                        cfg["resolution"],
                        cfg["width"],
                        cfg["depth"],
                        cfg["kernel_size"],
                        cfg["expand_ratio"],
                    )
                if teacher:
                    teacher.set_active_subnet(
                        cfg["resolution"],
                        cfg["width"],
                        cfg["depth"],
                        cfg["kernel_size"],
                        cfg["expand_ratio"],
                    )

            outputs, subnet_features, _ = model(samples)
            sub_loss = criterion(outputs, targets)
            if teacher:
                with torch.no_grad():
                    _, teacher_feature, _ = teacher(samples)
                sub_teach_loss = teacher_criteration(outputs, teacher_soft_logit)
                fkd_sub_loss = 0.0
                for i in range(len(subnet_features)):
                    fkd_sub_loss += teacher_criteration(outputs, teacher_soft_logit)
                fkd_sub_loss /= len(subnet_features)
                if not math.isnan(fkd_sub_loss) and not math.isnan(sub_teach_loss):
                    sub_loss += gamma * sub_teach_loss + beta * fkd_sub_loss
            sub_loss.backward()
        
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.TRAIN.CLIP_GRAD
            )
            grad_norm_meter.update(grad_norm.item())

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # Update learning rate scheduler
            lr_scheduler.step_update(epoch * num_steps + idx)

            # Update EMA
            if model_ema is not None:
                model_ema.update(model)
                
        batch_time.update(time.time() - start)
        start = time.time()


        # Logging
        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            eta_seconds = batch_time.avg * (num_steps - idx)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))

            log_message = (
                f"Epoch: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"ETA: {eta}\t"
                f"LR: {lr:.6f}\t"
                f"Data Time: {data_time.val:.4f} ({data_time.avg:.4f})\t"
                f"Batch Time: {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"Loss: {loss.item() * config.TRAIN.ACCUMULATION_STEPS:.4f}\t"
                f"Grad Norm: {grad_norm_meter.val:.4f}\t"
                f"Mem: {memory_used:.0f}MB"
            )
            if config.distributed:
                log_message = f"Rank: [{dist.get_rank()}]\t" + log_message
            logging.get_logger(__name__).info(log_message)

    logging.get_logger(__name__).info(
        f"Epoch [{epoch}] training takes {str(datetime.timedelta(seconds=int(time.time() - start)))}"
    )


def validate(config, train_loader, valid_loader, model, device, writer=None):
    """
    Validates the model on the validation dataset.
    """
    subnets_to_evaluate = {
        "attentive_nas_random_net": {},
        "attentive_nas_min_net": {},
        "attentive_nas_max_net": {},
    }

    criterion = nn.CrossEntropyLoss()

    nas_validate(
        subnets_to_evaluate,
        train_loader,
        valid_loader,
        model,
        criterion,
        config,
        logging.get_logger(__name__),
        bn_calibration=True,
        device=device,
        writer=writer,
    )


def throughput(data_loader, model, logger):
    """
    Measures the throughput of the model.
    """
    model.eval()
    with torch.no_grad():
        for idx, (images, _) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            batch_size = images.shape[0]

            # Warm-up
            for _ in range(50):
                model(images)
            torch.cuda.synchronize()

            logger.info("Throughput averaged over 30 iterations")
            start_time = time.time()
            for _ in range(30):
                model(images)
            torch.cuda.synchronize()
            end_time = time.time()

            throughput = (end_time - start_time) / (30 * batch_size)
            logger.info(
                f"Batch size {batch_size} throughput: {throughput:.6f} sec/sample"
            )
            break  # Only evaluate the first batch


def main():
    args, config = parse_option()

    # Set deterministic behavior for reproducibility
    cudnn.deterministic = True

    # Initialize distributed training if enabled
    if config.distributed:
        ngpus_per_node = int(
            os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count())
        )
        config.WORLD_SIZE = ngpus_per_node * int(os.environ.get("SLURM_NODES", 1))
        config.RANK = int(os.environ.get("SLURM_PROCID", 0))
        os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        mp.spawn(
            main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config, args)
        )
    else:
        config.defrost()
        ngpus_per_node = 1
        config.WORLD_SIZE = 1
        config.RANK = 0
        config.freeze()
        main_worker(0, ngpus_per_node, config, args)


if __name__ == "__main__":
    main()
