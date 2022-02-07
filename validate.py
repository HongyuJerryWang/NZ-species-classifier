import os
import sys
import math
import time
import timm
import torch
import pickle
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
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

# validation batch size
BATCH_SIZE = 32
# model size, choose from s, m, and l
MODEL_SIZE = 's'
# number of CPU data loading workers, should be not greater than the number of CPU threads
NUM_WORKERS = 6
# location of the checkpoint directory from training
CHECKPOINTS = 'checkpoints'
# intended save location of tensorboard log files
LOG_DIR = 'log'
# location of the auxiliary instance count file for accuracy binning
INSTANCE_COUNT = 'instance_count.pkl'

# initialise model architecture, loss function, and softmax function
model = timm.create_model(f'tf_efficientnetv2_{MODEL_SIZE}', pretrained=False, num_classes=11047).cuda()
criterion = nn.CrossEntropyLoss().cuda()
softmax = nn.Softmax(dim=-1).cuda()

# initialise preprocessing transformation and data pipeline
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
val_dataset = datasets.ImageFolder(
    'dataset/test',
    transforms.Compose([
        transforms.Resize(416 if MODEL_SIZE == 's' else 512),
        transforms.CenterCrop(384 if MODEL_SIZE == 's' else 480),
        transforms.ToTensor(),
        normalize,
    ])
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)

# locate saved checkpoints
checkpoints_dir = Path(CHECKPOINTS)

# initialise tensorboard log writers
writer_summary = SummaryWriter(f'{LOG_DIR}/summary')
writer_confidence = SummaryWriter(f'{LOG_DIR}/confidence')
writer_margin = SummaryWriter(f'{LOG_DIR}/margin')
writer_entropy = SummaryWriter(f'{LOG_DIR}/entropy_complement')

# initialise bins based on the number of instances in each class
instance_count = pickle.load(open(INSTANCE_COUNT, 'rb'))
class_bins = {v: (0 if instance_count[k] < 5 else (1 if instance_count[k] < 10 else (2 if instance_count[k] < 20 else (3 if instance_count[k] < 50 else 4)))) for k, v in val_dataset.class_to_idx.items()}

# specify abstention thresholds, with confidence (top-1 probability), margin (top-1 - top-2), or (complement of) entropy
confidence_thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]
margin_thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]
entropy_thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]

# track the best checkpoint
best_correct = -1
best_epoch = -1

# iterate through the checkpoints
for epoch in range(0, 285 if MODEL_SIZE == 's' else 505, 5):

    # load checkpoint
    state_dict = {
        (k[7:] if k[:7] == 'module.' else k) : v
        for k, v in
        torch.load(f'checkpoints/checkpoint_epoch{epoch}.pth')['model'].items()
    }
    model.load_state_dict(state_dict)
    del state_dict

    # prepare for validation
    model.eval()
    running_loss = 0.0
    processed_imgs = 0
    top_correct = [0 for _ in range(5)]
    processed_binned = [0 for _ in range(5)]
    correct_binned = [0 for _ in range(5)]
    confidence_processed_imgs = [0 for _ in confidence_thresholds]
    confidence_correct = [0 for _ in confidence_thresholds]
    margin_processed_imgs = [0 for _ in margin_thresholds]
    margin_correct = [0 for _ in margin_thresholds]
    entropy_processed_imgs = [0 for _ in entropy_thresholds]
    entropy_correct = [0 for _ in entropy_thresholds]

    with torch.no_grad():

        # iterate through mini-batches
        for step, data in enumerate(val_loader):

            # compute loss and predicted probabilities
            inputs = data[0].cuda()
            targets = data[1].cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            processed_imgs += targets.size(0)
            probabilities = softmax(outputs)

            # compute entropy from probabilities, with the entropy standardised to be between 0 and 1
            entropy = Categorical(probs=probabilities).entropy() / torch.log(torch.IntTensor([11047])).item()
            # compute top 5 classes and confidence
            confidence, predicted = torch.topk(probabilities, 5, 1)
            # compute margin
            confidence_diff = confidence.t()[0] - confidence.t()[1]
            # compute top-1 confidence
            confidence = confidence.t()[0]
            # compute number of correct predictions
            predicted = predicted.t()
            correct_labels = predicted.eq(targets.view(1, -1).expand_as(predicted))

            # aggregate number of passed instances and correct predictions based on each confidence abstention threshold
            for i, threshold in enumerate(confidence_thresholds):
                confidence_mask = (confidence >= threshold)
                confidence_processed_imgs[i] += confidence_mask.sum().item()
                confidence_correct[i] += (correct_labels[0] * confidence_mask).sum().item()

            # aggregate number of passed instances and correct predictions based on each margin abstention threshold
            for i, threshold in enumerate(margin_thresholds):
                margin_mask = (confidence_diff >= threshold)
                margin_processed_imgs[i] += margin_mask.sum().item()
                margin_correct[i] += (correct_labels[0] * margin_mask).sum().item()

            # aggregate number of passed instances and correct predictions based on each entropy abstention threshold
            for i, threshold in enumerate(entropy_thresholds):
                entropy_mask = (entropy <= (1.0 - threshold))
                entropy_processed_imgs[i] += entropy_mask.sum().item()
                entropy_correct[i] += (correct_labels[0] * entropy_mask).sum().item()

            # aggregate top-k correct predictions and binned correct predictions
            top_correct = [correct_labels[:k + 1].sum().item() + tc for k, tc in enumerate(top_correct)]
            targets_binned = [class_bins[target] for target in targets.cpu().tolist()]
            correct_list = correct_labels[0].cpu().tolist()
            for tb, cl in zip(targets_binned, correct_list):
                processed_binned[tb] += 1
                correct_binned[tb] += cl

        # update on validation progress
        print(f'Epoch {epoch} Top 5 accuracy', ' '.join([f'{100.0*tc/processed_imgs:3.3f}' for tc in top_correct]), flush=True)
        if top_correct[0] >= best_correct:
            best_correct = top_correct[0]
            best_epoch = epoch

    # save tensorboard logs of loss, binned accuracy, and top-k accuracy
    writer_summary.add_scalar('summary/loss', running_loss / (step + 1), epoch)
    for k, bin_range in enumerate(['1_4', '5_9', '10_19', '20_49', '50_']):
        writer_summary.add_scalar('summary/bin' + bin_range, 100.0 * correct_binned[k] / processed_binned[k], epoch)
    for k in range(5):
        writer_summary.add_scalar(f'summary/top{k+1:d}', 100.0 * top_correct[k] / processed_imgs, epoch)

    # log confidence-based abstention results, comparing across different thresholds or different epochs, as well as plotting accuracy against percentage of predictions made
    for i, threshold in enumerate(confidence_thresholds):
        writer_confidence.add_scalar(f'abstention_intra_epoch/predicted_epoch_{epoch:d}', 100 * confidence_processed_imgs[i] / processed_imgs, int(100 * threshold))
        writer_confidence.add_scalar(f'abstention_intra_epoch/acc_epoch_{epoch:d}', 0.0 if confidence_processed_imgs[i] == 0 else 100.0 * confidence_correct[i] / confidence_processed_imgs[i], int(100 * threshold))
        writer_confidence.add_scalar(f'abstention_intra_epoch/acc_vs_predicted_epoch_{epoch:d}', 0.0 if confidence_processed_imgs[i] == 0 else 100.0 * confidence_correct[i] / confidence_processed_imgs[i], round(100 * confidence_processed_imgs[i] / processed_imgs))
        writer_confidence.add_scalar(f'abstention_inter_epoch/predicted_threshold_{threshold:1.2f}', 100 * confidence_processed_imgs[i] / processed_imgs, epoch)
        writer_confidence.add_scalar(f'abstention_inter_epoch/acc_threshold_{threshold:1.2f}', 0.0 if confidence_processed_imgs[i] == 0 else 100.0 * confidence_correct[i] / confidence_processed_imgs[i], epoch)

    # log margin-based abstention results, comparing across different thresholds or different epochs, as well as plotting accuracy against percentage of predictions made
    for i, threshold in enumerate(margin_thresholds):
        writer_margin.add_scalar(f'abstention_intra_epoch/predicted_epoch_{epoch:d}', 100 * margin_processed_imgs[i] / processed_imgs, int(100 * threshold))
        writer_margin.add_scalar(f'abstention_intra_epoch/acc_epoch_{epoch:d}', 0.0 if margin_processed_imgs[i] == 0 else 100.0 * margin_correct[i] / margin_processed_imgs[i], int(100 * threshold))
        writer_margin.add_scalar(f'abstention_intra_epoch/acc_vs_predicted_epoch_{epoch:d}', 0.0 if margin_processed_imgs[i] == 0 else 100.0 * margin_correct[i] / margin_processed_imgs[i], round(100 * margin_processed_imgs[i] / processed_imgs))
        writer_margin.add_scalar(f'abstention_inter_epoch/predicted_threshold_{threshold:1.2f}', 100 * margin_processed_imgs[i] / processed_imgs, epoch)
        writer_margin.add_scalar(f'abstention_inter_epoch/acc_threshold_{threshold:1.2f}', 0.0 if margin_processed_imgs[i] == 0 else 100.0 * margin_correct[i] / margin_processed_imgs[i], epoch)

    # log entropy-based abstention results, comparing across different thresholds or different epochs, as well as plotting accuracy against percentage of predictions made
    for i, threshold in enumerate(entropy_thresholds):
        writer_entropy.add_scalar(f'abstention_intra_epoch/predicted_epoch_{epoch:d}', 100 * entropy_processed_imgs[i] / processed_imgs, int(100 * threshold))
        writer_entropy.add_scalar(f'abstention_intra_epoch/acc_epoch_{epoch:d}', 0.0 if entropy_processed_imgs[i] == 0 else 100.0 * entropy_correct[i] / entropy_processed_imgs[i], int(100 * threshold))
        writer_entropy.add_scalar(f'abstention_intra_epoch/acc_vs_predicted_epoch_{epoch:d}', 0.0 if entropy_processed_imgs[i] == 0 else 100.0 * entropy_correct[i] / entropy_processed_imgs[i], round(100 * entropy_processed_imgs[i] / processed_imgs))
        writer_entropy.add_scalar(f'abstention_inter_epoch/predicted_threshold_{threshold:1.2f}', 100 * entropy_processed_imgs[i] / processed_imgs, epoch)
        writer_entropy.add_scalar(f'abstention_inter_epoch/acc_threshold_{threshold:1.2f}', 0.0 if entropy_processed_imgs[i] == 0 else 100.0 * entropy_correct[i] / entropy_processed_imgs[i], epoch)

# print the best checkpoint based on top-1 accuracy
print(f'Best epoch {best_epoch} Accuracy {100.0*best_correct/processed_imgs:3.3f}')
