# Automatic Segmentation of Right Ventricle in Cardiac MRIs
## Machine Learning Nanodegree Capstone Report
###### Akshar Kumar
###### Feb. 2018

## I. Definition

### Project Overview
Segmentation of the right ventricle (RV) in patients is a crucial step in diagnosing debilitating diseases. RV segmentation is used when diagnosing diseases such as pulmonary hypertension, coronary heart disease, and cardiomypathies, among others (REF1). Currently, the accepted gold standard for evaluating RV volumes is a manual contour by a trained physician on cardiac MRI images. The physician will perform contours on consecutive images resulting in a set of 2D contours that when combined, can form a 3D volume of the ventricle. Since there is usually a large number of images to contour, the overall process of segmentation becomes very lengthy. Each image can take anywhere from 10-15 minutes. Aside from the large time cost, there are some difficulties that can make the examination task more subjective:
- fuzziness of borders due to blood flow
- presence of wall irregularities 
- complex shape of the RV - it can appear triangular when viewed longitudinally and cresent-shaped when viewed along the short axis
- variability in cine MRI equipment, institutions, and populations
- noise associated with cine MRI images [REF1, REF2]

Because of these difficulties, the process is prone to intra- and inter-observer variability [REF1]. The purpose of this project was to simplify these complex pain points and provide a method to automatically segment the RV in cardiac MRI images. Such an automated method would decrease segmentation time and improve consistency of results between physicians. The automated model would then assist the physician in making better diagnoses for the patient.

The dataset used for this project was provided in the Right Ventricle Segmentation Competition conducted in 2012 [FOOT1]. For more information about the dataset, see Section II [SECTION2].

### Problem Statement
In this project, we will apply a deep learning model (Convolutional Neural Network) to automatically segment the right ventricle in cardiac MRI images. Segmentation of the right ventricle is useful in characterizing the ejection fraction of the heart. We will compare the accuracy of the model to segmentation performed by physicians with years of experience. These manual contours will be what we call the ground truth. This problem was presented as a computer vision challenge at the International Conference on Medical Image Computing and Computer Assisted Intervention in October 2012 [FOOT1]. 

The solution we will implement is a convolutional neural network based on the U-Net architecture [REF5]. The U-Net architecture is a popular architecture used for biomedical image segmentation. In this architecture, there is a downsampling path that follows the same structure as a generic CNN. There are convolutional layers followed by max pooling layers, with each step halving the overall image space. The architecture then introduces an upsampling path, which is needed to create a segmentation map of the input image with similar dimensions. In this upsampling path, the image size increases, while the number of channels decrease. In this way we get a segmentation map that will tell us whether or not a pixel is part of our region of interest. [Include figure?]. We also used this architecture because the authors were able to successfully train a model with only 30 images of labeled data. Since we also have few labeled images, we believe this architecture will provide us with a good start for our model.

The tasks that we have to complete during this project can be broken down into three larger categories:

1. Preprocess Data
   - Clean the data and gather only the labeled images
   - Explore the data by calculating the image histogram
   - See how histogram equalization affects training and the data.
2. Build and train the model
   - Create a model similar to the U-Net architecture proposed [REF5]
   - Split data set into training and validation data
   - Augment the training data.
   - Train on training data with validation.
   - Tune hyperparameters: dropout, batch normalization, number of epochs, batch size, etc.
3. Test the model
   - Predict contours from the testing data.
   - Convert the segmentation map to list of points in contour shape.
   - Submit these predictions to moderators for evaluation.

### Metrics

For evaluating our model, we used the Dice coefficient. The Dice coefficient is a measure of the overlap between two contours. The coefficient varies from 0 to 1, with a value of 0 indicating no overlap between two contours. Meanwhile, a value of 1 indicates a perfect overlap between two contours. We compared our automated contours to a manual contour performed by an expert physician using the following equation: 

D(A,M) = 2*((AnM)/(A+M))

as desribed by [REF2]. Where D represents the Dice coefficient, A represents the area of the automated contour, and M represents the area of the manual contour performed by the expert. 


## II. Analysis
### Data Exploration
The dataset used for this project was provided in the Right Ventricle Segmentation Competition conducted in 2012. The training dataset contains images from 48 patients: 16 training cases, 32 testing cases. For each patient there were a total of 200-280 imges. The images provided in each case were  2D cine images with approximately 8-12 continuous images spanning a single cardiac cycle for each patient along the short axis view. The cardiac images had been zoomed and cropped to a 216x256 pixel region of interest. The labels provided were manual RV segmentations of images. These segmentations were performed by an expert. The processing time per patient was around 15 minutes. There were a total of 243 labeled images in our training dataset. The testing set contained 514 labeled images without manual contours. We submitted our contour predictions on the test images to the moderators for final evaluation. 

### Exploratory Visualization
### Algorithms and Techniques
### Benchmark

## Methodologies
### Data Preprocessing
### Implementation
### Refinement

## IV. Results
### Model Evaluation and Validation
### Justification

## V. Conclusion
### Free-Form Visualization
### Reflection
### Improvement
