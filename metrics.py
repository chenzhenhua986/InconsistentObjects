import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
#from scipy.special import softmax
from scipy.special import expit
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
np.set_printoptions(threshold=sys.maxsize)


def center_error(output, label, class_num=3):
    b, c, w, h = output.shape
    max_idx = np.argmax(output, axis=1)
    correct_num=0
    for i in range(b):
      true_label=sum(sum((label[i,:,:]).astype(int))) / w / h
      pred_list=[] 
      for j in range(class_num):
        v = sum(sum((max_idx[i, :, :] == j).astype(int)))
        pred_list.append(v)
      pred_label=np.argmax(pred_list)

      #print('ground truth label vs predicted label: ' + str(true_label) + ' ' + str(pred_label))
      if true_label == pred_label:
        correct_num += 1
    acc = correct_num / b * 100.0
    return acc

def AUC(output, label):
    """ Calculates the area under the ROC curve of the given outputs. All of the
    outputs in the neutral region of the label (see labels.py) are ignored when
    calculating the AUC.
    Why I'm using SkLearn's AUC: https://github.com/pytorch/tnt/issues/54

    Args:
        output: (np.ndarray) The output of the network with dimension [Bx1xHxW]
        label: (np.ndarray) The labels with dimension [BxHxWx2]
    """
    b = output.shape[0]
    output = output.reshape(b, -1)
    mask = label[:, :, :, 1].reshape(b, -1)
    label = label[:, :, :, 0].reshape(b, -1)
    total_auc = 0
    for i in range(b):
        print(i)
        total_auc += roc_auc_score(label[i], output[i], sample_weight=mask[i])
        #total_auc += confusion_matrix(label[i], output[i], sample_weight=mask[i])
    return total_auc/b



# The dictionary containing all the available metrics.
METRICS = {
    'accuracy': {'fcn': center_error}
}
