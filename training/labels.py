import numpy as np

def create_BCELogit_loss_label(label_size, same):
    """
      Create multi-class labels according to 'same'
    """
    label = np.zeros([label_size, label_size]).astype(np.float32)
    if int(same) == 0:
      label[:,:] = int(0)

    if int(same) == 1:
      label[:, :] = int(1)

    if int(same) == -1:
      label[:, :] = int(2)

    return label
