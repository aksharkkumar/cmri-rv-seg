import os,glob,sys
import os,glob,sys
import numpy as np
import matplotlib.pyplot as plt

import cv2
from src import data, unet

class Predictor(object):
    def __init__(self,data_dir,weight_file):
        glob_search = os.path.join(data_dir,"patient*")
        self.patient_dirs=sorted(glob.glob(glob_search))
        weight_endo_file = os.path.join("saved_models/endo_models",weight_file)
        #weight_epi_file = os.path.join("saved_models/epi_models",weight_file)
        images, _ = self.load_images(self.patient_dirs[0])
        _, height, width, channels = images.shape
        #self.o_model = unet.UNet().get_unet(height=height,width=width,channels=channels,features=32,steps=3)
        self.i_model = unet.UNet().get_unet(height=height,width=width,channels=channels,features=32,steps=3)
        #self.o_model.load_weights(weight_dir)
        self.i_model.load_weights(weight_endo_file)

    def make_predictions(self,out_dir):
        for path in self.patient_dirs:
            images,p_ids,rotated = self.load_images(path)
            o_predictions=[]
            i_predictions=[]
            for image in images:
                o_mask_pred = self.o_model.predict(image[None,:,:,:])
                i_mask_pred = self.i_model.predict(image[None,:,:,:])
                o_predictions.append((image[:,:,0],o_mask_pred[0,:,:,1]))
                i_predictions.append((image[:,:,0],i_mask_pred[0,:,:,1]))
            self.save_predictions(o_predictions,p_ids,rotated,"o",out_dir)
            self.save_predictions(i_predictions,p_ids,rotated,"i",out_dir)

        
        return o_predictions, i_predictions

    def make_predictions_one(self,data_dir,num_imgs):
        images, _ = self.load_images(data_dir)
        size = len(images)
        images_trunc = images[:num_imgs] # only look at a few images
        self.normalize(images_trunc)
        o_predictions = []
        i_predictions = []
        for image in images_trunc:
            #o_mask_pred = self.o_model.predict(image[None,:,:,:])
            i_mask_pred = self.i_model.predict(image[None,:,:,:])
            #o_predictions.append((image[:,:,0],o_mask_pred))
            i_predictions.append((image[:,:,0],i_mask_pred))
        return o_predictions,i_predictions

    def load_images(self,path):
        img_data_obj = data.ImageData(path)
        imgs = img_data_obj.labeled_images
        print(len(imgs))
        images=np.asarray(imgs)[:,:,:,None].astype('float64')
        #self.normalize(images, axis=(1,2))
        return images, img_data_obj.rotated
    
    def normalize(x, epsilon=1e-7, axis=(1,2)):
        x -= np.mean(x, axis=axis, keepdims=True)
        x /= np.std(x, axis=axis, keepdims=True) + epsilon
    

    def save_predictions(self,predictions,p_ids,rotated,class_type,out_dir):
        for(image,mask),p_id in zip(predictions,p_ids):
            filename = p_id + class_type + "contour-auto.txt"
            outpath = os.path.join(out_dir,filename)
            print(filename)
            contour = self.generate_contours(mask)
            if rotated:
                height, width = image.shape
                x, y = contour.T
                x, y = height - y, x
                contour = np.vstack((x,y)).T
        np.savetxt(outpath,contour,fmt='%i',delimiter=' ')
    
    def generate_contours(self,mask):
        mask_image = np.where(mask>0.5,255,0).astype('uint8')
        im2, coords, hierarchy = cv2.findContours(mask_image, cv2.RETER_LIST, cv2.CHAIN_APPROX_NONE)
        coords = np.squeeze(coords[0],axis=(1,))
        coords = np.append(coords,coords[:1],axis=0)
        return coords

    def create_prediction_images(self):
        return self