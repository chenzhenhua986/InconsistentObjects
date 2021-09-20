import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

def center_error(output, label, class_num=3):
    output = output.cpu().data.numpy()
    label = label.cpu().data.numpy()
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

      if true_label == pred_label:
        correct_num += 1

    err = 1.0 - correct_num / b

    return err

def BCELogit_Loss(input, target, weight=None, size_average=False):
    """ custom Cross-entropy loss. The output has a shape of w * h * class_number. If w=h=1, it can be replaced with nn.CrossEntropyLoss()
    """
    n, c, h, w = input.size()
    output = F.log_softmax(input, dim=1)
    target = target.long() 
    criterion = nn.NLLLoss()
    loss = criterion(output, target)

    loss /= n
    err = center_error(input, target)
    alpha = 1
    return alpha*loss + (1-alpha)*err




