import numpy as np
from keras.layers import Input, Conv2D, Conv2DTranspose, Cropping2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras import backend as K


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
        for i in reversed(range(steps)):
            features //= 2
            layer = Conv2DTranspose(filters=features,kernel_size=2,strides=2)(layer)
            crop_copy = self.crop(layer,copies[i])
            layer.Concatenate()( [layer, crop_copy] )

            layer = Conv2D(filters=features,kernel_size=3,activation='relu',padding='valid')(layer)
            layer = Conv2D(filters=features,kernel_size=3,activation='relu',padding='valid')(layer)

        
        outputs = Conv2D(filters=2,kernel_size=1,activation='softmax')(layer)

        return Model(inputs=inputs,outputs=outputs)




    def crop(self,layer,conv_copy):
        _,layer_height,layer_width = K.int_shape(layer)
        _,cc_height,cc_width = K.int_shape(conv_copy)

        crop_height = cc_height - layer_height
        crop_width = cc_width - layer_width


        if crop_height == 0 and crop_width == 0:
            copy = layer
        else :
            cropping = ((crop_height // 2, crop_height - crop_height//2), (crop_width, crop_width - crop_width//2))
            copy = Cropping2D(cropping=cropping)(layer)

        return copy
        