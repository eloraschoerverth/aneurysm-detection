import torch
from torch import nn
from scipy import signal
from scipy import spatial

## https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5

class Metric(nn.Module):
    def forward(self, prediction, label):
        prediction = prediction.reshape(-1)
        label = label.reshape(-1)
        tp = torch.sum(label*prediction)
        fn = torch.sum(label) - tp
        fp = torch.sum(prediction) - tp
        tn = label.shape[0] - (torch.sum(label) + fp)
        return tp, tn, fp, fn

class TverskyLoss(Metric):
    
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, prediction, label):
        tp, tn, fp, fn = super().forward(prediction, label)
        tversky = tp /(tp + self.alpha * fn + (1-self.alpha) * fp)
        return 1 - tversky
    

class DiceLoss(Metric):
    
    def forward(self, prediction, label):
        tp, tn, fp, fn = super().forward(prediction, label)
        dice =  2*tp/(2*tp + fn + fp)
        return 1 - dice

class Sensitivity(Metric):
    
    def forward(self, prediction, label):
        tp, tn, fp, fn = super().forward(prediction, label)
        return tp /(tp + fn)

class Specificity(Metric):

    def forward(self, prediction, label):
        tp, tn, fp, fn = super().forward(prediction, label)
        return tn /(tn + fp)
    
class Accuracy(Metric):

    def forward(self, prediction, label):
        tp, tn, fp, fn = super().forward(prediction, label)
        return (tn+tp)/(tp+tn+fp+fn)
    
class Jaccard(Metric):
    
    def forward(self, prediction, label):
        tp, tn, fp, fn = super().forward(prediction, label)
        return (tp) / (tp + fp + fn)

##################### Segmentation metrics ##########################


#We need flattened pred/labels for these metrics
class SegmentationMetric(nn.Module):  
    def forward(self, prediction, label):
        prediction = prediction.reshape(-1)
        label = label.reshape(-1)
        return prediction, label

class StdDev(SegmentationMetric):
  
    def forward(self, prediction, label):
        pred, lab = super().forward(prediction, label)
        return (pred - lab).std()
    
class Bias(SegmentationMetric):
    def forward(self, prediction, label):
        pred, lab = super().forward(prediction, label)
        #absolute difference
        diff = torch.abs(torch.sum(pred - lab))
        #amount of non-zero elements in label mask
        sum_label = torch.sum(label)
        return 1/sum_label*diff

class AvgDist(SegmentationMetric):
  
    def forward(self, prediction, label):
        pred, lab = super().forward(prediction, label)
        avg = spatial.distance.cdist(pred.numpy(), label.numpy(), metric='euclidean')
        return sum(sum(avg)) / (avg.shape[0] * avg.shape[1])
