import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model



class UNet(object):
    def __init__(self, img_rows, img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def unet(self):
        '''
            Creates the UNet conv model. 
            Downsampling : repeat downsample 4 times
                1. Conv 3x3 n filters with ReLU
                2. Conv 3x3 n filters with ReLU
                3. Max pool 2x2

            Conv 3x3 with ReLU
            Conv 3x3 with ReLU

            Upsampling : repeat upsample 4 times
                1. Conv 3x3 n filters with ReLU
                2. Conv 3x3 n filters with ReLU
                3. Up-conv 2x2
                
            Conv 1x1 => output segmentation map
        '''
        
        