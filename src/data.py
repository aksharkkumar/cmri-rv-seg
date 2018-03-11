import os, glob, re
import numpy as np
import dicom

from PIL import Image, ImageDraw

class ImageData(object):
    def __init__(self, dir):
        self.dir = os.path.normpath(dir) # normalize path from "\" to "/" to match file structure
        search = os.path.join(dir, "P*list.txt")
        search_results = glob.glob(search)
        self.contour_files = search_results[0]

        match = re.search("P(..)list.txt", self.contour_files)
        self.index = int(match.group(1))

        self.images = []
        self.dicoms = []
        self.endo_contours = []
        self.epi_contours = []
        self.endo_masks = []
        self.epi_masks = []
        self.width = 256
        self.height = 216
        self.rotated = False
        # load contours first, so we can build a set of labeled images 
        # => then grab only those patient images
        self.images, self.dicoms = self.load_patient_images()
        try:
            self.load_patient_masks()
        except FileNotFoundError:
            pass
        
        
        
    @property
    def labeled_images(self):
        return [self.images[val] for val in self.labels]
    def load_patient_images(self):
        dcm_path = os.path.join(self.dir, "*dicom/*.dcm")
        dcms = sorted(glob.glob(dcm_path))
        images = []
        dicoms = []
        for dcm in dcms:
            # print(dcm)
            #match = re.search(".*(........).dcm",dcm)
            #dcm_key = match.group(1)
            # print(dcm_key)
            # print(dcm_key)
            ds = dicom.read_file(dcm)
            img = self.rotate_image(ds.pixel_array)
            images.append( img )
            dicoms.append( ds )
        return images, dicoms

    def load_patient_masks(self):
        '''
            We are given the contours as [X,Y] pixel pairs. However to
            compare the two areas we need to determine a "mask" that 
            will convert the image to a binary image. The area outlined
            by the contours will be white and the background will 
            be black.

            Based on the Matlab evaluation code provided with dataset.
        '''
        # read the paths of contur text files into list
        with open(self.contour_files, 'r') as f:
            files = f.readlines()
        files = [x.strip() for x in files] # need to remove \n from end of lines as they are paths
        
        i_contour_files = files[0::2] # inner contours start at 0 index and every other index
        o_contour_files = files[1::2] # outer contours start at 1 index and every other index

        i_contour_files = [path.replace("\\","/") for path in i_contour_files]
        o_contour_files = [path.replace("\\","/") for path in o_contour_files]

        
        
        # for i_contour_file in i_contour_files:
        #     match = re.search("patient../(........)-.contour", i_contour_file)
        #     img_label = match.group(1)
        #     self.img_keys.add(img_label)
        
        
        self.labels = []
        for i_contour_file, o_contour_file in zip(i_contour_files, o_contour_files):
            # build set of labeled images => not all images are labeled
            match = re.search("P..-(....)-.contour",i_contour_file)
            p_idx = int(match.group(1))
            #print(p_idx)
            self.labels.append(p_idx)
            i_x, i_y = self.read_contour_files(i_contour_file)
            o_x, o_y = self.read_contour_files(o_contour_file)
            self.endo_contours.append(  (i_x,i_y) )
            self.epi_contours.append( (o_x,o_y) )
            self.endo_masks.append( self.create_masks(i_x, i_y) )
            self.epi_masks.append( self.create_masks(o_x, o_y) )



    def read_contour_files(self,filepath):
        match = re.search("patient../(.*)", filepath)
        path = os.path.join(self.dir, match.group(1))
        # print(path)
        x, y = np.loadtxt(path).T
        if self.rotated:
            x, y = y, self.height - x
        return x, y

    def create_masks(self, x, y):
        image = Image.new("L", (self.width,self.height))
        polygon = list(zip(x,y))
        ImageDraw.Draw(image).polygon(polygon,fill="white",outline="white")
        return 255 * np.array(image)

    def rotate_image(self, image):
        img_height, img_width = image.shape
        if img_width < img_height:
            self.rotated = True
            return np.rot90(image)
        else:
            return image

