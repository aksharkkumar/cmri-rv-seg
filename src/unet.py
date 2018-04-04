import numpy as np
from keras.layers import Input, Conv2D, Conv2DTranspose, Cropping2D, MaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.layers import Dropout, Flatten, Dense, Concatenate, Activation, BatchNormalization
from keras.models import Model
from keras import backend as K


class UNet(object):

    def get_unet(self, height, width, channels, features=64, steps=4, dropout=0.0, padding='same'):
        '''
            Creates the UNet conv model. 
            Downsampling : repeat downsample 4 times
                1. Conv 3x3 n filters with ReLU
                2. Conv 3x3 n filters with ReLU
                3. Max pool 2x2

            Conv 3x3 with ReLU
            Conv 3x3 with ReLU

            Upsampling : repeat upsample 4 times
                1. Up-conv 2x2
                2. Conv 3x3 n filters with ReLU
                3. Conv 3x3 n filters with ReLU
                

            Conv 1x1 => output segmentation map
        '''
        layer = Input(shape=(height,width,channels))
        inputs = layer
        copies = []
        # downsampling block
        for i in range(steps):
            layer = Conv2D(filters=features,kernel_size=3,padding=padding)(layer)
            #layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(dropout)(layer)

            layer = Conv2D(filters=features,kernel_size=3, padding=padding)(layer)
            #layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(dropout)(layer)

            copies.append(layer)
            layer = MaxPooling2D(pool_size=(2, 2))(layer)
            
            features *= 2

        layer = Conv2D(filters=features,kernel_size=3,padding=padding)(layer)
        #layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(dropout)(layer)

        layer = Conv2D(filters=features,kernel_size=3,padding=padding)(layer)
        #layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(dropout)(layer)

        # upsampling block
        for i in reversed(range(steps)):
            features //= 2
            layer = Conv2DTranspose(filters=features,kernel_size=2,strides=2)(layer)
            crop_copy = self.crop(layer,copies[i])
            layer = Concatenate()( [layer, crop_copy] )

            layer = Conv2D(filters=features,kernel_size=3,padding=padding)(layer)
            #layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(dropout)(layer)

            layer = Conv2D(filters=features,kernel_size=3,padding=padding)(layer)
            #layer = BatchNormalization()(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(dropout)(layer)

        
        layer = Conv2D(filters=2,kernel_size=1)(layer)
        
        layer = Lambda(lambda x: x/1.0)(layer)
        outputs = Activation('softmax')(layer)
        
        return Model(inputs=inputs,outputs=outputs)




    def crop(self,layer,conv_copy):
        '''
            Function used to crop the copy of the downsampling layer for use with
            each corresponding upsampling layer. Cropping is necessary because 
            padding was not used when downsampling, so image size has changed.
        '''
        _, layer_height, layer_width, _ = K.int_shape(layer)
        _, cc_height, cc_width, _ = K.int_shape(conv_copy)

        crop_height = cc_height - layer_height
        crop_width = cc_width - layer_width


        if crop_height == 0 and crop_width == 0:
            copy = layer
        else :
            cropping = ((crop_height // 2, crop_height - crop_height//2), (crop_width // 2, crop_width - crop_width//2))
            copy = Cropping2D(cropping=cropping)(conv_copy)

        return copy
    
    