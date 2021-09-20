import os
from math import sqrt
import numpy as np
from imageio import imread
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

from training.crops_train import crop_img, resize_and_pad
from utils.exceptions import IncompatibleImagenetStructure
from training.train_utils import get_annotations, check_folder_tree
from training.labels import create_BCELogit_loss_label as BCELoss

class Consistency(Dataset):
    def __init__(self, img_dir="/N/u/chen478/fall2018/apu_project/Pytorch-SiamFC/consistency_dataset/", transforms=ToTensor(),
                 reference_size=127, search_size=255, final_size=33,
                 #reference_size=127, search_size=255, final_size=1,
                 label_fcn=BCELoss,
                 cxt_margin=0.5, single_label=False, img_read_fcn=imread,
                 resize_fcn=[], metadata_file=None, save_metadata=None):
        self.reference_size = reference_size
        self.search_size = search_size
        self.cxt_margin = cxt_margin
        self.final_size = final_size
        self.transforms = transforms
        self.label_fcn = label_fcn
        if single_label:
            self.label = self.label_fcn(self.final_size, self.pos_thr,
                                        self.neg_thr,
                                        upscale_factor=self.upscale_factor)
        else:
            self.label = None
        self.img_read = img_read_fcn
        self.resize_fcn = resize_fcn

        self.root = '/N/u/chen478/fall2018/apu_project/Pytorch-SiamFC/consis_dataset/'
        f = open(self.root + 'scripts/train_labels.txt')
        self.lines = f.readlines()
        self.cursor = 0 
    
    def ref_context_size(self, h, w):
        margin_size = self.cxt_margin*(w + h)
        ref_size = sqrt((w + margin_size) * (h + margin_size))
        # make sur ref_size is an odd number
        ref_size = (ref_size//2)*2 + 1
        return int(ref_size)

    def preprocess_sample(self, p1, p2, label):
        reference_frame_path = self.root + p1.split('/')[8] + '/' + p1.split('/')[9] + '/' + p1.split('/')[9] + '_image-0001.png'
        ref_annot = dict()
        with open(p1, "r") as f:
          first_line = f.readline()
          ref_annot['xmin'] = int(first_line.split(' ')[0])
          ref_annot['ymin'] = int(first_line.split(' ')[1])
          ref_annot['xmax'] = int(first_line.split(' ')[2])
          ref_annot['ymax'] = int(first_line.split(' ')[3])

        ref_w = (ref_annot['xmax'] - ref_annot['xmin'])/2
        ref_h = (ref_annot['ymax'] - ref_annot['ymin'])/2
        ref_ctx_size = self.ref_context_size(ref_h, ref_w)
        ref_cx = (ref_annot['xmax']+ref_annot['xmin'])/2
        ref_cy = (ref_annot['ymax']+ref_annot['ymin'])/2
        ref_frame = self.img_read(reference_frame_path)
        ref_frame = np.float32(ref_frame)
        ref_frame, pad_amounts_ref = crop_img(ref_frame, ref_cy, ref_cx, ref_ctx_size)
        try:
            ref_frame = resize_and_pad(ref_frame, self.reference_size, pad_amounts_ref,
                                       reg_s=ref_ctx_size, use_avg=True, resize_fcn=self.resize_fcn)
        except AssertionError:
            print('Fail Ref: ', reference_frame_path)
            raise

        search_frame_path = self.root + p2.split('/')[8] + '/' + p2.split('/')[9] + '/' + p2.split('/')[9] + '_image-0001.png'
        srch_annot = dict()
        with open(p1, "r") as f:
          first_line = f.readline()
          srch_annot['xmin'] = int(first_line.split(' ')[0])
          srch_annot['ymin'] = int(first_line.split(' ')[1])
          srch_annot['xmax'] = int(first_line.split(' ')[2])
          srch_annot['ymax'] = int(first_line.split(' ')[3])

        srch_ctx_size = ref_ctx_size * self.search_size / self.reference_size
        srch_ctx_size = (srch_ctx_size//2)*2 + 1

        srch_cx = (srch_annot['xmax'] + srch_annot['xmin'])/2
        srch_cy = (srch_annot['ymax'] + srch_annot['ymin'])/2
        srch_frame = self.img_read(search_frame_path)
        srch_frame = np.float32(srch_frame)
        srch_h = (srch_annot['xmax'] - srch_annot['xmin'])
        srch_w = (srch_annot['ymax'] - srch_annot['ymin'])
        srch_frame, pad_amounts_srch = crop_img(srch_frame, srch_cy, srch_cx, srch_ctx_size)
        try:
            srch_frame = resize_and_pad(srch_frame, self.search_size, pad_amounts_srch,
                                        reg_s=srch_ctx_size, use_avg=True, resize_fcn=self.resize_fcn)
        except AssertionError:
            print('Fail Search: ', search_frame_path)
            raise


        label_transform = self.label_fcn(self.final_size, label)

        seq_idx = 0

        ref_frame = self.transforms(ref_frame)
        srch_frame = self.transforms(srch_frame)

        out_dict = {'ref_frame': ref_frame, 'srch_frame': srch_frame, 'label': label_transform, 'seq_idx': seq_idx}
        return out_dict


    def __getitem__(self, idx):
        max_idx = len(self.lines) 
        if self.cursor >= max_idx:
          self.cursor = 0
        line = self.lines[self.cursor].strip().split(' ')

        p1 = line[0] 
        p2 = line[1]
        label = line[2]
        self.cursor += 1
        if self.cursor >= max_idx:
          self.cursor = 0
        return self.preprocess_sample(p1, p2, label)

    def __len__(self):
        return len(self.lines)

class Consistency_val(Consistency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.root = '/N/u/chen478/fall2018/apu_project/Pytorch-SiamFC/consis_dataset/'
        f = open(self.root + 'scripts/chuhua_yuchen_141821_141655.txt')
        #f = open(self.root + 'scripts/jianxin_152628_152728.txt')
        #f = open(self.root + 'scripts/violet_xiwen_151011_155354.txt')
        #f = open(self.root + 'scripts/tmp.txt')
        #f = open(self.root + 'scripts/martin_hua.txt')
        self.lines = f.readlines()
        self.cursor = 0 
         

    def preprocess_sample(self, p1, p2, label):
        reference_frame_path = self.root + p1.split('/')[8] + '/' + p1.split('/')[9] + '/' + p1.split('/')[9] + '_image-' + p1.split('/')[11].split('_')[0] + '.png'
        ref_annot = dict()
        with open(p1, "r") as f:
          first_line = f.readline()
          ref_annot['xmin'] = int(first_line.split(' ')[0])
          ref_annot['ymin'] = int(first_line.split(' ')[1])
          ref_annot['xmax'] = int(first_line.split(' ')[2])
          ref_annot['ymax'] = int(first_line.split(' ')[3])
 
        ref_w = (ref_annot['xmax'] - ref_annot['xmin'])/2
        ref_h = (ref_annot['ymax'] - ref_annot['ymin'])/2
        ref_ctx_size = self.ref_context_size(ref_h, ref_w)
        ref_cx = (ref_annot['xmax']+ref_annot['xmin'])/2
        ref_cy = (ref_annot['ymax']+ref_annot['ymin'])/2
        ref_frame = self.img_read(reference_frame_path)
        ref_frame = np.float32(ref_frame)
        ref_frame, pad_amounts_ref = crop_img(ref_frame, ref_cy, ref_cx, ref_ctx_size)
        try:
            ref_frame = resize_and_pad(ref_frame, self.reference_size, pad_amounts_ref,
                                       reg_s=ref_ctx_size, use_avg=True, resize_fcn=self.resize_fcn)
        except AssertionError:
            print('Fail Ref: ', reference_frame_path)
            raise
        search_frame_path = self.root + p2.split('/')[8] + '/' + p2.split('/')[9] + '/' + p2.split('/')[9] + '_image-' + p2.split('/')[11].split('_')[0] + '.png'
        srch_annot = dict()
        with open(p1, "r") as f:
          first_line = f.readline()
          srch_annot['xmin'] = int(first_line.split(' ')[0])
          srch_annot['ymin'] = int(first_line.split(' ')[1])
          srch_annot['xmax'] = int(first_line.split(' ')[2])
          srch_annot['ymax'] = int(first_line.split(' ')[3])

        srch_ctx_size = ref_ctx_size * self.search_size / self.reference_size
        srch_ctx_size = (srch_ctx_size//2)*2 + 1

        srch_cx = (srch_annot['xmax'] + srch_annot['xmin'])/2
        srch_cy = (srch_annot['ymax'] + srch_annot['ymin'])/2
        srch_frame = self.img_read(search_frame_path)
        srch_frame = np.float32(srch_frame)
        srch_h = (srch_annot['xmax'] - srch_annot['xmin'])
        srch_w = (srch_annot['ymax'] - srch_annot['ymin'])
        srch_frame, pad_amounts_srch = crop_img(srch_frame, srch_cy, srch_cx, srch_ctx_size)
        try:
            srch_frame = resize_and_pad(srch_frame, self.search_size, pad_amounts_srch,
                                        reg_s=srch_ctx_size, use_avg=True, resize_fcn=self.resize_fcn)
        except AssertionError:
            print('Fail Search: ', search_frame_path)
            raise

        label_transform = self.label_fcn(self.final_size, label)

        seq_idx = 0

        ref_frame = self.transforms(ref_frame)
        srch_frame = self.transforms(srch_frame)
        out_dict = {'ref_frame': ref_frame, 'srch_frame': srch_frame, 'label': label_transform, 'seq_idx': seq_idx}
        return out_dict


    def __getitem__(self, idx):
        max_idx = len(self.lines) 
        if self.cursor >= max_idx:
          self.cursor = 0
        line = self.lines[self.cursor].strip().split(' ')

        p1 = line[0] 
        p2 = line[1]
        label = line[2]
        self.cursor += 1
        if self.cursor >= max_idx:
          self.cursor = 0
        return self.preprocess_sample(p1, p2, label)
        
        
    def __len__(self):
        return len(self.lines)
