from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from calendar import c

import os
import torch
import argparse
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, iou, tensor_to_PIL, get_box_data
from PIL import Image, ImageDraw
from tqdm import tqdm
import sklearn
import sklearn.metrics


# hyper-parameters
# ------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--lr',
    default=0.0001,
    type=float,
    help='Learning rate'
)
parser.add_argument(
    '--lr-decay-steps',
    default=15000,
    type=int,
    help='Interval at which the lr is decayed'
)
parser.add_argument(
    '--lr-decay',
    default=0.5,
    type=float,
    help='Decay rate of lr'
)
parser.add_argument(
    '--momentum',
    default=0.9,
    type=float,
    help='Momentum of optimizer'
)
parser.add_argument(
    '--weight-decay',
    default=0.0005,
    type=float,
    help='Weight decay'
)
parser.add_argument(
    '--epochs',
    default=5,
    type=int,
    help='Number of epochs'
)
parser.add_argument(
    '--val-interval',
    default=5000,
    type=int,
    help='Interval at which to perform validation'
)
parser.add_argument(
    '--disp-interval',
    default=10,
    type=int,
    help='Interval at which to perform visualization'
)
parser.add_argument(
    '--use-wandb',
    default=True,
    type=bool,
    help='Flag to enable visualization'
)
# ------------

# Set random seed
rand_seed = 1024
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

# Set output directory
output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class_id_to_label = dict(enumerate(VOCDataset().CLASS_NAMES))
logged = set()

def calculate_map(res):
    """
    Calculate the mAP for classification.
    """
    # TODO (Q2.3): Calculate mAP on test set.
    # Feel free to write necessary function parameters.
    # get predictions and ground truths
    pred = res[0]
    pred = sorted(pred, key=lambda x: x[0], reverse=True)
    gt = res[1]
    # iterate all ground truths to get number of gt_bboxes for each class
    allgtbboxes = [0] * 20
    for i in range(len(gt)):
        for j in range(len(gt[i][2])):
            allgtbboxes[gt[i][2][j]] += 1
    # total for each class
    AP = [0] * 20
    TP = [0] * 20
    FP = [0] * 20
    # visited set for each ground truth bboxes list, used for if one gt_bbox is match,
    # it cannot be matched again
    visited = [set() for _ in range(len(gt))]
    # precision and recall list for AP (AUC) calculation for each class
    precision = [[] for _ in range(20)]
    recall = [[] for _ in range(20)]
    # iterate prediction bounding boxes to calculate precision and recall
    for _, box, class_num, iter in pred:
        gt_boxes = gt[iter][1]
        gt_class_list = gt[iter][2]
        assert gt[iter][0] == iter
        assert len(gt_boxes) == len(gt_class_list)
        # if not more gt bboxes to match a prediction, this prediction is a false positive
        if len(gt_boxes) == 0 or len(gt_boxes) == len(visited[iter]):
            FP[class_num] += 1
        else:
            match = False
            # iterate gt bboxes of the image
            for i in range(len(gt_boxes)):
                if i in visited[iter]:
                    continue
                # prediction is true positive if label is matched and iou exceeds threshold
                if class_num == gt_class_list[i] and iou(box, gt_boxes[i]) > 0.45:
                    TP[class_num] += 1
                    visited[iter].add(i)
                    match = True
                    break
            # prediction is not matched to any gt bboxes and is regarded a a false positive
            if not match:
                FP[class_num] += 1

        # add precision and recall to list
        precision[class_num].append(TP[class_num] / (TP[class_num] + FP[class_num]))
        recall[class_num].append(TP[class_num] / allgtbboxes[class_num])
    # using sklearn auc function to calculate AP for each class
    for i in range(20):
        AP[i] = sklearn.metrics.auc(recall[i], precision[i])
    return AP


def test_model(model, val_loader=None, thresh=0.05, epoch=0, args=None):
    """
    Tests the networks and visualizes the detections
    :param thresh: Confidence threshold
    """
    res = [[], []]
    cnt = 0
    with torch.no_grad():
        cnt = 0
        for iter, data in tqdm(enumerate(val_loader), total=len(VOCDataset(split='test'))):
            # one batch = data for one image
            image = data['image']
            target = data['label']
            rois = data['rois']
            gt_boxes = data['gt_boxes']
            gt_class_list = data['gt_classes']

            # TODO (Q2.3): perform forward pass, compute cls_probs
            image = image.cuda()
            target = target.cuda()
            rois = torch.squeeze(rois)
            rois = rois.cuda()
            cls_prob = model(image, [rois], target)


            # TODO (Q2.3): Iterate over each class (follow comments)
            labels = []
            bboxes = []
            for class_num in range(20):
                # get valid rois and cls_scores based on thresh
                # use NMS to get boxes and scores
                boxes, scores = nms(rois.cpu(), cls_prob[:, class_num].cpu(), thresh)
                if boxes and scores:
                    for i in range(len(boxes)):
                        res[0].append((scores[i], boxes[i], class_num, iter))
                        if args.use_wandb and (epoch == 0 or epoch == 4) and (cnt < 10 or iter in logged):
                            labels.append(class_num)
                            bboxes.append(boxes[i].detach().numpy().astype(float))
            res[1].append([iter, gt_boxes, gt_class_list])

            # TODO (Q2.3): visualize bounding box predictions when required
            if args.use_wandb:
                if epoch == 0 and cnt < 10:
                    if labels and bboxes:
                        img = wandb.Image(tensor_to_PIL(torch.squeeze(image).cpu()), boxes={
                            "predictions": {
                                "box_data": get_box_data(labels, bboxes),
                                "class_labels": class_id_to_label,
                                },
                            })
                        wandb.log({f'{iter}th image with bbox': img})
                        logged.add(iter)
                        cnt += 1
                elif epoch == 4 and iter in logged:
                    if labels and bboxes:
                        img = wandb.Image(tensor_to_PIL(torch.squeeze(image).cpu()), boxes={
                            "predictions": {
                                "box_data": get_box_data(labels, bboxes),
                                "class_labels": class_id_to_label,
                                },
                            })
                        wandb.log({f'{iter}th image with bbox': img})

        return calculate_map(res)


def train_model(model, train_loader=None, val_loader=None, optimizer=None, args=None):
    """
    Trains the network, runs evaluation and visualizes the detections
    """
    # Initialize training variables
    # LR Step scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=args.lr_decay_steps,
                                                gamma=args.lr_decay)
    train_loss = 0
    step_cnt = 0
    for epoch in range(args.epochs):
        for iter, data in tqdm(enumerate(train_loader), total=len(VOCDataset(split='trainval'))):
            # TODO (Q2.2): get one batch and perform forward pass
            # one batch = data for one image
            image = data['image']
            target = data['label']
            rois = data['rois']

            # TODO (Q2.2): perform forward pass
            # take care that proposal values should be in pixels
            # Convert inputs to cuda if training on GPU
            image = image.cuda()
            target = target.cuda()
            rois = rois.cuda()
            rois = torch.squeeze(rois)
            model(image, [rois], target)


            # backward pass and update
            loss = model.loss
            train_loss += loss.item()
            step_cnt += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # TODO (Q2.2): evaluate the model every N iterations (N defined in handout)
            # Add wandb logging wherever necessary
            if iter % args.val_interval == 0 and iter != 0:
                model.eval()
                ap = test_model(model, val_loader, 0.05, epoch, args)
                model.train()

            # TODO (Q2.4): Perform all visualizations here
            # The intervals for different things are defined in the handout
            if iter % 500 == 0:
                if args.use_wandb:
                    wandb.log({'train/loss': train_loss / step_cnt})
        # TODO (Q2.4): Plot class-wise APs
        if args.use_wandb:
            wandb.log({'test/mAP': np.mean(ap),
                       'test/AP_0': ap[0],
                       'test/AP_1': ap[1],
                       'test/AP_2': ap[2],
                       'test/AP_3': ap[3],
                       'test/AP_4': ap[4]})
    print(f'final AP: {ap}')
    print(f'final mAP: {np.mean(ap)}')


def main():
    """
    Creates dataloaders, network, and calls the trainer
    """
    args = parser.parse_args()
    # TODO (Q2.2): Load datasets and create dataloaders
    # Initialize wandb logger
    if args.use_wandb:
        wandb.init(project="vlr-hw1")
    train_dataset = VOCDataset(split='trainval',
                               image_size=512)
    val_dataset = VOCDataset(split='test',
                             image_size=512)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,   # batchsize is one for this implementation
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True)

    # Create network and initialize
    net = WSDDN(classes=train_dataset.CLASS_NAMES)
    print(net)

    if os.path.exists('pretrained_alexnet.pkl'):
        pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
    else:
        pret_net = model_zoo.load_url(
            'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
        pkl.dump(pret_net,
        open('pretrained_alexnet.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)
    own_state = net.state_dict()

    for name, param in pret_net.items():
        print(name)
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name].copy_(param)
            print('Copied {}'.format(name))
        except:
            print('Did not find {}'.format(name))
            continue

    # Move model to GPU and set train mode
    net.load_state_dict(own_state)
    net.cuda()
    net.train()

    # TODO (Q2.2): Freeze AlexNet layers since we are loading a pretrained model
    for param in net.features.parameters():
        param.requires_grad = False

    # TODO (Q2.2): Create optimizer only for network parameters that are trainable
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Training
    train_model(net, train_loader, val_loader, optimizer, args)

main()