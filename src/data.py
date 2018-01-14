import os, glob, re
import numpy as np
import dicom


class ImageData(object):
    def __init__(self, dir):
        self.dir = dir
        search = os.path.join(dir, "P*list.txt")
        search_results = glob.glob(search)
        self.contour_files = search_results[0]
        self.load_patient_images()
        self.load_masks()
    
    def load_patient_images(self):
        dcm_path = os.path.join(self.dir, "*dicom/*.dcm")
        dcms = glob.glob(dcm_path)
        images = []
        dicoms = []
        for dcm in dcms:
            ds = dicom.read_file(dcm)
            images.append(ds.pixel_array)
            dicoms.append(ds)
        return images, dicoms

    def load_patient_masks(self):
        # read the paths of contur text files into list
        with open(self.contour_files) as f:
            files = f.readlines()
        files = [x.strip() for x in files] # need to remove \n from end of lines as they are paths
        
        i_contour_files = files[0::2] # inner contours start at 0 index and every other index
        o_contour_files = files[1::2] # outer contours start at 1 index and every other index

        i_contour_files = [path.replace("\\","/") for path in i_contour_files]
        o_contour_files = [path.replace("\\","/") for path in o_contour_files]

        # build set of labeled images => not all images are labeled #TODO

    def load_masks(self):
        return self


def load_images(dir):
    patient_dirs_path = os.path.join(dir, "patient*")
    
    images = []
    dicoms = []
    epi_contours = []
    endo_contours = []
    for patient_image_dir in patient_dirs_path:
        imd = ImageData(patient_image_dir)
        images, dicoms = imd.load_patient_images()
    return images
