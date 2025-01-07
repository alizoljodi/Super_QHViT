import random
import os
import time
import argparse
import datetime
import numpy as np
import warnings
import sys
import math
import logging
from copy import deepcopy
from collections import defaultdict
import gc
from data import build_hf_tiny_imagenet_loader, build_loader
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
import torch.multiprocessing as mp
import torch.profiler
import torch.nn.functional as F
import torch.nn as nn
from torch.amp import autocast

logging.getLogger("PIL").setLevel(logging.WARNING)
# Suppress all warnings
warnings.filterwarnings("ignore")
# from torch.amp  import GradScaler
from torch.utils.tensorboard import SummaryWriter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
import timm as timm
from timm.utils import ModelEma
from misc.config import get_config
import misc.attentive_nas_eval as attentive_nas_eval
import models
from torch.nn.parallel import DistributedDataParallel as DDP
from misc.lr_scheduler import build_scheduler
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

from misc.loss_ops import AdaptiveLossSoft

logger = logging.get_logger(__name__)
grad_norms = []
glob_epoch = 0


def parse_option():
    parser = argparse.ArgumentParser(
        "Super-HQViT Training and Evaluation", add_help=False
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=r"/home/ali/Downloads/TinyIMGNT_HQViT_NAS/HQViT_NAS_Tiny_imgnt/configs/cfg.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=f"/home/ali/Downloads/TinyIMGNT_HQViT_NAS/test",
        help="root dir for models and logs",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Enable distribured training",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--resume", type=str, help="resume path")
    parser.add_argument(
        "--fp_teacher_dir", type=str, help="Path to full-precision supernet weights"
    )
    parser.add_argument(
        "--accumulation-steps", type=int, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )
    args, unparsed = parser.parse_known_args()

    # setup the work dir
    args.output = args.working_dir
    writer = SummaryWriter(os.path.join(args.output, "log"))
    # args.tag = args.workflow_run_id or args.tag  # override settings

    config = get_config(args)
    return args, config


def _setup_worker_env(ngpus_per_node, config):
    config.defrost()
    if config.distributed:
        config.LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        config.RANK = int(os.environ["RANK"])
        config.WORLD_SIZE = int(os.environ["WORLD_SIZE"])
        if True:
            print(
                f"""
    =============================================
    Rank: {config.RANK}
    Local rank: {config.LOCAL_RANK}
    World size: {config.WORLD_SIZE}
    Master addres: {os.environ["MASTER_ADDR"]}
    Master port: {os.environ["MASTER_PORT"]}
    =============================================
            """
            )
        dist.init_process_group(
            backend="nccl", rank=config.RANK, world_size=config.WORLD_SIZE
        )
        torch.cuda.set_device(f"cuda:{config.LOCAL_RANK}")
    else:
        gpu = 0
        config.RANK = gpu
        config.WORLD_SIZE = 1
        config.gpu = gpu
        config.LOCAL_RANK = gpu
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:10001",
            rank=config.RANK,
            world_size=1,
        )
        torch.cuda.set_device(f"cuda:{config.LOCAL_RANK}")
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    assert (
        dist.get_world_size() == config.WORLD_SIZE
    ), "DDP is not properply initialized."
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = (
        config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    )
    linear_scaled_warmup_lr = (
        config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    )
    linear_scaled_min_lr = (
        config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    )
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = (
            linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        )
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # Setup logging format.
    logging.setup_logging(
        os.path.join(config.OUTPUT, "stdout.log"),
        "w",
    )

    # backup the config
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())


def main_worker(rank, ngpus_per_node, config):
    _setup_worker_env(ngpus_per_node, config)
    device = torch.device(f"cuda:{config.LOCAL_RANK}")
    (
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
        mixup_fn,
        # random_erasing,
    ) = build_loader(config)
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = models.model_factory.create_model(
        config, parent="model", full_precision=False
    )
    print(model)
    model.cuda()

    logger.info(str(model))
    model_ema = ModelEma(model, decay=0.99985, device=device, resume="")

    optimizer = build_optimizer(config, model)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], broadcast_buffers=False, find_unused_parameters=True
    )
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.0:
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.MODEL.RESUME and True == False:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, logger
        )
        model_ema.ema = deepcopy(model_without_ddp)
        if config.EVAL_MODE:
            return
    logger.info("Start training")
    start_time = time.time()
    supernet_gn = defaultdict(float)

    for epoch in range(100):
        glob_epoch = epoch

        supernet_gn = train_one_epoch(
            config=config,
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            mixup_fn=mixup_fn,
            lr_scheduler=lr_scheduler,
            teacher=None,
            model_ema=model_ema,
            supernet_gradnorm=supernet_gn,
            device=device,
            random_erasing=None,
        )
        if dist.get_rank() == 0 and (
            epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)
        ):
            save_checkpoint(
                config,
                epoch,
                model,
                max_accuracy,
                optimizer,
                lr_scheduler,
                logger,
                model,
            )
        # validate(config, data_loader_train, data_loader_val, model,device)
        if epoch % 1 == 0:
            validate(config, data_loader_train, data_loader_val, model, device)
    glob_epoch = 0
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    destroy_process_group()


def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
    teacher=None,
    model_ema=None,
    supernet_gradnorm=None,
    drop=False,
    device="",
    random_erasing=None,
):
    model.train()

    num_steps = len(data_loader)
    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    super_loss_meter = AverageMeter()
    grad_norms = []

    norm_meter = AverageMeter()

    ce_criterion = nn.CrossEntropyLoss()

    start = time.time()
    end = time.time()

    cfgs = []

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        data_time.update(time.time() - end)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if random_erasing:
            samples = random_erasing(samples)

        cfg = model.module.sample_max_subnet(quantized=True)

        output, model_features, block_names = model(samples)

        super_loss = criterion(output, targets)

        if not math.isfinite(super_loss.item()):
            pass

        super_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        grad_norms.append(grad_norm)
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        if model_ema is not None:
            model_ema.update(model)

        norm_meter.update(grad_norm)
        super_loss_meter.update(super_loss)

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            if config.distributed:
                logger.info(
                    f"Rank: [{config.LOCAL_RANK}]\t"
                    f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                    f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                    f"data {data_time.val:.4f} ({data_time.avg:.4f})\t"
                    f"batch {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                    f"ce_loss_meter {super_loss_meter.val:.4f} ({super_loss_meter.avg:.4f})\t"
                    f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                    f"mem {memory_used:.0f}MB"
                )
            # prof.step()
            else:
                logger.info(
                    f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                    f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                    f"data {data_time.val:.4f} ({data_time.avg:.4f})\t"
                    f"batch {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                    f"ce_loss_meter {super_loss_meter.val:.4f} ({super_loss_meter.avg:.4f})\t"
                    f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                    f"mem {memory_used:.0f}MB"
                )

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )

    return supernet_gradnorm


@torch.no_grad()
def validate(config, train_loader, valid_loader, model, device, writer=None):
    subnets_to_be_evaluated = {
        "attentive_nas_random_net": {},
        "attentive_nas_min_net": {},
        "attentive_nas_max_net": {},
    }

    criterion = nn.CrossEntropyLoss()

    attentive_nas_eval.validate(
        subnets_to_be_evaluated,
        train_loader,
        valid_loader,
        model,
        criterion,
        config,
        logger,
        bn_calibration=True,
        device=device,
        writer=writer,
    )


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(
            f"batch_size {batch_size} throughput {(tic2 - tic1)/ (30 * batch_size) }"
        )
        return


if __name__ == "__main__":
    _, config = parse_option()

    cudnn.deterministic = True
    ngpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"]) if config.distributed else 1
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    # mp.spawn(main_worker, args=(ngpus_per_node, config))
    main_worker(None, ngpus_per_node, config)
