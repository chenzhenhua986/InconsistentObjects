import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, zeros_, normal_
from torchvision import datasets, models, transforms
from torchsummary import summary


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        #model = models.resnet152(pretrained=False).cuda()
        model = models.mobilenet_v3_large(pretrained=True).cuda()
        self.fully_conv = nn.Sequential(*(list(model.children())[:-1]))
        summary(self.fully_conv, (3, 224, 224))
 
    def forward(self, x):
        output = self.fully_conv(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net, corr_map_size=33):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.match_batchnorm = nn.BatchNorm2d(3)

        #self.upscale = upscale
        self.corr_map_size = corr_map_size
        #self.stride = stride
        #self.upsc_size = (self.corr_map_size-1)*self.stride + 1
        #if upscale:
        #    self.upscale_factor = 1
        #else:
        #    self.upscale_factor = self.stride

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): The reference patch of dimensions [B, C, H, W].
                Usually the shape is [8, 3, 127, 127].
            x2 (torch.Tensor): The search region image of dimensions
                [B, C, H', W']. Usually the shape is [8, 3, 255, 255].
        Returns:
            match_map (torch.Tensor): The score map for the pair. For the usual
                input shapes, the output shape is [8, 1, 33, 33].
        """
        embedding_reference = self.embedding_net(x1)
        embedding_search = self.embedding_net(x2)
        match_map = self.match_corr(embedding_reference, embedding_search)
        #match_map = nn
        return match_map

    def get_embedding(self, x):
        return self.embedding_net(x)


    def match_corr(self, embed_ref, embed_srch):
        """ Matches the two embeddings using the correlation layer. As per usual
        it expects input tensors of the form [B, C, H, W].

        Args:
            embed_ref: (torch.Tensor) The embedding of the reference image, or
                the template of reference (the average of many embeddings for
                example).
            embed_srch: (torch.Tensor) The embedding of the search image.

        Returns:
            match_map: (torch.Tensor) The correlation between
        """
        #print('shape of matching paris: ')
        b, c, h, w = embed_srch.shape
        b1, c1, h1, w1 = embed_ref.shape
        # Here the correlation layer is implemented using a trick with the
        # conv2d function using groups in order to do the correlation with
        # batch dimension. Basically we concatenate each element of the batch
        # in the channel dimension for the search image (making it
        # [1 x (B.C) x H' x W']) and setting the number of groups to the size of
        # the batch. This grouped convolution/correlation is equivalent to a
        # correlation between the two images, though it is not obvious.

        #match_map = F.conv2d(embed_srch.view(1, b * c, h, w), embed_ref, groups=b)
        #match_map = match_map.permute(1, 0, 2, 3)
        #print(match_map.shape)

        srch_reshape = embed_srch.view(1, b * c, h, w)
        ref_reshape = embed_ref.view(b1 * 3, b1 * c1 // (b1 * 3), h1, w1)
        match_map = F.conv2d(srch_reshape, ref_reshape, groups=b1 * 3)
        b2, c2, h2, w2 = match_map.shape
        match_map = match_map.view(b, c2//b, h2, w2)
        match_map = self.match_batchnorm(match_map)
        #if self.upscale:
        #    match_map = F.interpolate(match_map, self.upsc_size, mode='bilinear',
        #                              align_corners=False)
        return match_map

