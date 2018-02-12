# Automatic Segmentation of Right Ventricle in Cardiac MRIs
## Machine Learning Nanodegree Capstone Report
###### Akshar Kumar

## I. Definition

### Project Overview
Segmentation of the right ventricle (RV) in patients is a crucial step in diagnosing debilitating diseases. RV segmentation is used when diagnosing diseases such as pulmonary hypertension, coronary heart disease, and cardiomypathies, among others (REF1). Currently, the accepted gold standard for evaluating RV volumes is a manual contour by a trained physician on cardiac MRI images. The physician will perform contours on consecutive images resulting in a set of 2D contours that when combined, can form a 3D volume of the ventricle. Since there is usually a large number of images to contour, the overall process of segmentation becomes very lengthy. Each image can take anywhere from 10-15 minutes. Aside from the large time cost, there are some difficulties that can make the examination task more subjective:
- fuzziness of borders due to blood flow
- presence of wall irregularities 
- complex shape of the RV - it can appear triangular when viewed longitudinally and cresent-shaped when viewed along the short axis
- variability in cine MRI equipment, institutions, and populations
- noise associated with cine MRI images [REF1, REF2]

Because of these difficulties, the process is prone to intra- and inter-observer variability [REF1]. The purpose of this project was to simplify these complex pain points and provide a method to automatically segment the RV in cardiac MRI images. Such an automated method would decrease segmentation time and improve consistency of results between physicians. The automated model would then assist the physician in making better diagnoses for the patient.

The dataset used for this project was provided in the Right Ventricle Segmentation Competition conducted in 2012. The training dataset contains images from 48 patients: 16 training cases, 32 testing cases. For each patient there were a total of 200-280 imges. The images provided in each case were  2D cine images with approximately 8-12 continuous images spanning a single cardiac cycle for each patient along the short axis view. The cardiac images have been zoomed and cropped to a 216x256 pixel region of interest. The labels provided were manual RV segmentations of images. There were a total of 243 labeled images in our training dataset. The testing set contained 514 labeled images without manual contours. We submitted our contour predictions on the test images to the moderators for final evaluation. 

### Problem Statement
In this project, we will apply a deep learning model (Convolutional Neural Network) to automatically segment the right ventricle in cardiac MRI images. Segmentation of the right ventricle is useful in characterizing the ejection fraction of the heart. We will compare the accuracy of the model to segmentation performed by physicians with years of experience. These manual contours will be what we call the ground truth. This problem was presented as a computer vision challenge at the International Conference on Medical Image Computing and Computer Assisted Intervention in October 2012 [FOOT1]. 

The solution we will implement is a convolutional neural network based on the U-Net architecture [REF5]. 

### Metrics

For evaluating our model, we used the Dice coefficient. The Dice coefficient is a measure of the overlap between two contours. The coefficient varies from 0 to 1, with a value of 0 indicating no overlap between two contours. Meanwhile, a value of 1 indicates a perfect overlap between two contours. We compared our automated contours to a manual contour performed by an expert physician using the following equation: 

D(A,M) = 2*((AnM)/(A+M))

as desribed by [REF2]. Where D represents the Dice coefficient, A represents the area of the automated contour, and M represents the area of the manual contour performed by the expert. 



