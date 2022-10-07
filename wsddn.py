from cProfile import label
import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.ops import roi_pool, roi_align


class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        # TODO (Q2.1): Define the WSDDN model
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )
        self.roi_pool = roi_pool
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.score_fc = nn.Linear(4096, self.n_classes)
        self.bbox_fc = nn.Linear(4096, self.n_classes)

        # loss
        self.cross_entropy = None

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):


        # TODO (Q2.1): Use image and rois as input
        # compute cls_prob which are N_roi X 20 scores
        x = self.features(image)
        x = self.roi_pool(x, rois, 6, 1.0 / 16 * 512)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        score = self.score_fc(x)
        score = F.softmax(score, 1)
        bbox = self.bbox_fc(x)
        bbox = F.softmax(bbox, 0)
        cls_prob = score * bbox


        if self.training:
            label_vec = gt_vec.view(self.n_classes)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        return cls_prob

    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        # TODO (Q2.1): Compute the appropriate loss using the cls_prob
        # that is the output of forward()
        # Checkout forward() to see how it is called
        loss = nn.BCELoss(reduction='sum')(torch.clamp(torch.sum(cls_prob, 0), 0, 1), label_vec)

        return loss
