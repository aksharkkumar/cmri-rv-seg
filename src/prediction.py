import os,glob,sys
import os,glob,sys
import numpy as np
import matplotlib.pyplot as plt

import cv2
from src import data, unet

class Predictor(object):
    def make_predictions(self,data_dir):
        glob_search = os.path.join(data_dir,"patient*")
        patient_dirs = sorted(glob.glob(glob_search))

        images, _, _ = self.load_images(patient_dirs[0])
        _, height, width, channels = images.shape
        print("Building models...")
        o_model = unet.UNet().get_unet(height=height,width=width,channels=channels,features=32,steps=3)
        i_model = unet.UNet().get_unet(height=height,width=width,channels=channels,features=32,steps=3)

        o_model.load_weights('notebooks/saved_models/endo_models/weightsNoDrop.hdf5')
        i_model.load_weights('notebooks/saved_models/epi_models/weightsNoDrop.hdf5')

        for path in patient_dirs:
            images,p_ids,rotated = self.load_images(path)
            o_predictions=[]
            i_predictions=[]
            for image in images:
                o_mask_pred = o_model.predict(image[None,:,:,:])
                i_mask_pred = i_model.predict(image[None,:,:,:])
                o_predictions.append((image[:,:,0],o_mask_pred[0,:,:,1]))
                i_predictions.append((image[:,:,0],i_mask_pred[0,:,:,1]))
        return o_predictions, i_predictions



    def load_images(self,path):
        img_data_obj = data.ImageData(path)
        images=np.asarray(img_data_obj.images.values,dtype='float64')[:,:,:,None]
        return images, img_data_obj.images.keys, img_data_obj.rotated
    
    
    def save_predictions(self,out_dir):
        return self
    def create_prediction_images(self):
        return self