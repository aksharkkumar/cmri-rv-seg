import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model



class UNet(object):
    def __init__(self,image):
        self.height,self.width = image.shape

    def unet(self, input, features=64, steps=4):
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
        layer = Input((self.height,self.width,1))
        copies = []

        for i in range(steps):
            layer = Conv2D(filters=features,kernel_size=3,activation='relu',padding='valid')(layer)
            layer = Conv2D(filters=features,kernel_size=3,activation='relu',padding='valid')(layer)
            layer = MaxPooling2D(pool_size=(2, 2))(layer)
            copies.append(layer)
            features *= 2

        layer = Conv2D(filters=features,kernel_size=3,activation='relu',padding='valid')(layer)
        layer = Conv2D(filters=features,kernel_size=3,activation='relu',padding='valid')(layer)

        #TODO implement upsampling portion


        