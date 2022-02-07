import os
import sys
import math
import time
import timm
import torch
import argparse

import numpy as np
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pathlib import Path
from torch.optim import RMSprop
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

# number of GPUs for data-parallel training
WORLD_SIZE = 4

# batch size on each GPU
BATCH_SIZE = 96

# model size, choose from s, m, and l
MODEL_SIZE = 'm'

# number of data loading worker processes, should ideally be no greater than the number of available threads
NUM_WORKERS = 16

# number of epochs divided into fine-tuning stages as the model is gradually unfrozen
EPOCHS = [10, 20, 30, 40, 50, 60, 70] if MODEL_SIZE == 's' else [25, 25, 50, 50, 75, 75, 100, 100]

# the starting epoch, use a none-zero number to resume from a specific checkpoint
START_EPOCH = 0

# directory where the checkpoints should be saved
checkpoints_dir = Path('checkpoints')
checkpoints_dir.mkdir(exist_ok = True)

# child process for each GPU training slice
def train(rank, args):

    # network configuration
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=WORLD_SIZE, rank=rank)
    # activating benchmark mode seems to make training slightly faster
    torch.backends.cudnn.benchmark = True
    print('GPU', rank, 'initialised', flush=True)

    # create model on GPU
    model = timm.create_model(f'tf_efficientnetv2_{MODEL_SIZE}_in21k', pretrained=True, num_classes=11047).to(rank)
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    torch.cuda.set_device(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    criterion = nn.CrossEntropyLoss().to(rank)
    print('GPU', rank, 'model created', flush=True)

    # data preprocessing and loading
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_dataset = datasets.ImageFolder(
        '/Scratch/repository/hw168/2021A/NZ-Species-Dataset/sanitised/merged_train',
        transforms.Compose([
            transforms.RandomResizedCrop(300 if MODEL_SIZE == 's' else 384),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=WORLD_SIZE, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, sampler=train_sampler)
    print('GPU', rank, 'loader ready', flush=True)

    # save initial checkpoint and initialise log writer
    if rank == 0 and START_EPOCH == 0:
        writer = SummaryWriter(str(checkpoints_dir / str(rank)))
        current_time = time.time()
        cp_path = checkpoints_dir / ("checkpoint_epoch0.pth")
        torch.save({
            'epoch': 0,
            'model': model.state_dict(),
            'optim': optimizer.state_dict()
        }, str(cp_path))
        print(f"Saved checkpoint to {str(cp_path)}")
    dist.barrier()
    # set up mixed-precision training
    scaler = amp.GradScaler()
    steps_per_epoch = len(train_loader)
    current_epoch = 0
    print('GPU', rank, 'training started', flush=True)

    # iterate through each fine-tuning stage
    for stage_i, epoch_per_stage in enumerate(EPOCHS):
        # unfreeze only the classifier layer at the zero-th stage
        if stage_i == 0:
            model.requires_grad_(False)
            model.module.classifier.requires_grad_(True)
        # unfreeze every component at the last stage
        elif stage_i == len(EPOCHS) - 1:
            model.module.requires_grad_(True)
        # unfreeze a block at every middle stage, starting from the last block
        else:
            # a few layers between the last block and the classifier need to be unfrozen along with the last block
            if stage_i = 1:
                model.module.bn2.requires_grad_(True)
                model.module.conv_head.requires_grad_(True)
            model.module.blocks[-stage_i].requires_grad_(True)

        # record stage-beginning epoch
        current_epoch_static = current_epoch

        # the resume point hasn't come yet, skip forward
        if current_epoch + epoch_per_stage <= START_EPOCH:
            current_epoch += epoch_per_stage
            continue

        # the resume point has been dealt with in one of the previous stages, proceed as normal
        elif current_epoch > START_EPOCH:
            optimizer = RMSprop(params = filter(lambda p: p.requires_grad, model.parameters()), lr = 0.000001 * WORLD_SIZE * BATCH_SIZE / 16, alpha=0.9, eps=1e-08, weight_decay=1e-5, momentum=0.9)
            lr_manager = LambdaLR(optimizer, lambda step: (0.99 ** (float(current_epoch_static) + float(step) / float(steps_per_epoch))))

        # iterate through epochs
        for epoch in range(current_epoch, current_epoch + epoch_per_stage):

            # the resume point is in this stage, but isn't here yet, iterate forward
            if epoch < START_EPOCH:
                current_epoch = epoch + 1
                continue
            # the resume point is here
            elif epoch == START_EPOCH:
                # initialise optimiser
                optimizer = RMSprop(params = filter(lambda p: p.requires_grad, model.parameters()), lr = 0.000001 * WORLD_SIZE * BATCH_SIZE / 16, alpha=0.9, eps=1e-08, weight_decay=1e-5, momentum=0.9)
                # load checkpoint
                if epoch > 0:
                    checkpoint = torch.load(f'checkpoints/checkpoint_epoch{START_EPOCH:d}.pth', map_location = map_location)
                    model.load_state_dict(checkpoint['model'])
                    # load optimiser if not resuming from beginning of a stage
                    if epoch > current_epoch_static:
                        optimizer.load_state_dict(checkpoint['optim'])
                    del checkpoint
                # initialise learning rate manager
                lr_manager = LambdaLR(optimizer, lambda step: 0.99 ** (START_EPOCH + float(step) / float(steps_per_epoch)))

            # set model to train mode and set data loading sampler to the correct epoch
            model.train()
            train_sampler.set_epoch(epoch)

            # iterate through mini-batches
            for step, data in enumerate(train_loader):
                # get input and correct labels
                inputs = data[0].cuda(rank, non_blocking=True)
                targets = data[1].cuda(rank, non_blocking=True)
                # clear gradients
                optimizer.zero_grad()
                # mixed-precision forward propagation
                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                # mixed-precision back propagation
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                lr_manager.step()
                scaler.update()
                # print some logs to tell user training is ongoing
                if rank == 0:
                    print(f"Epoch {epoch} Step {step} GPU {rank} Loss {loss.item():6.4f} Latency {time.time()-current_time:4.3f} LR {optimizer.param_groups[0]['lr']:1.8f}", flush=True)
                    current_time = time.time()
                    # log learning rate and loss values for tensorboard plotting
                    if step % 100 == 99:
                        writer.add_scalar('train/loss', loss.item(), steps_per_epoch * epoch + step)
                        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], steps_per_epoch * epoch + step)

            # save a checkpoint every five epochs
            if epoch % 5 == 4:
                if rank == 0:
                    cp_path = checkpoints_dir / ("checkpoint_epoch" + str(epoch + 1) + ".pth")
                    torch.save({
                        'epoch': epoch + 1,
                        'model': model.state_dict(),
                        'optim': optimizer.state_dict()
                    }, str(cp_path))
                    print(f"Saved checkpoint to {str(cp_path)}")
                dist.barrier()

            # update the current epoch
            current_epoch = epoch + 1

# main process
if __name__ == '__main__':
    # set up and start child processes, one for each GPU
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    mp.spawn(train, nprocs = WORLD_SIZE, args=(args,))
