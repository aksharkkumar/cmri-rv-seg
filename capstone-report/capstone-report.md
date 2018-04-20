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
The dataset used for this project was provided in the Right Ventricle Segmentation Competition conducted in 2012. The training dataset contains images from 48 patients: 16 training cases, 32 testing cases. For each patient there were a total of 200-280 imges. The images provided in each case were  2D cine images with approximately 8-12 continuous images spanning a single cardiac cycle for each patient along the short axis view. The cardiac images had been zoomed and cropped to a 256x216 pixel region of interest. The labels provided were manual RV segmentations of images. These segmentations were performed by an expert. The processing time per patient was around 15 minutes. There were a total of 243 labeled images in our training dataset. The testing set contained 514 labeled images without manual contours. We submitted our contour predictions on the test images to the moderators for final evaluation. [Include Figures of mri images with contours overlayed]. Upon exploring the image data, we noticed that data for some patients was rotated 90 degrees, so instead of being 256x216, the images were 216x256. When loading the images into memory to use for training, we rotated all the images to be 256x216. 

The labels provided were epicardium (outer wall) and endocardium (inner wall) contours. The set of patients images that were contoured were listed in a text file. Using this file, we only loaded images into memory that had corresponding contours. The contours were given as text files with [X,Y] pairs corresponding to the pixel in the image that was part of the contour. When loading the images, we also made sure to rotate contours for images that were not of the 256x216 size. The contours were converted to binary masks using OpenCV. 

One of the main issues with the training set is the amount of images given. Because the labeled images available are so few, we performed some data augmentation before training. We introduced random rotations, zooms, and transformations. [Include figure showing the transformations]. The augmentation was performed using the ImageDataGenerator API available in Keras. 

### Exploratory Visualization
- Image specific histograms?
### Algorithms and Techniques
- Histogram equalization?
### Benchmark
The benchmark model we compared our model to was the fully convolutional neural network created by Phi Vu Tran for the RVSC competition [REF2]. This model is a 15-layer deep fully convolutional neural network (FCN). The FCN comprised of 15 stacked convolutional layers and three max-pooling layers. Each of the convolution layers used a Rectified Linear Unit (ReLU) as the activation function. The total number of parameters was around 11 million. The output of this model was a heat map that predicted the class membership of each pixel in the images. In order to compare the proposed model to this benchmark model, we will use the Dice coefficient, as described in Section I. For our benchmark model, the Dice coefficient for endocardium contours was 0.84(0.21) with standard deviation in parentheses. The epicardium contours Dice was 0.86(0.20). These values were the average Dice coefficients from teh two test sets provided in the challenge. 

We will also compare our model to human level performance. The average Dice coefficient for a human is estimated to be around 0.90(0.10) [REF1]. 

## Methodologies
### Data Preprocessing
The dataset we used in this project was relatively clean. The only issue that had to be taken care of was rotating images to the same height and width. Most of the images in the dataset were 256x216, however there a few images for patients that were 216x256, or rotated 90 degrees. We rotated the images while loading them into memory. In addition to rotating the images, we also had to rotate the contours. This was also done when the image masks were being created from the contour data. 

To solve the main issue within our data, it's small size, we used data augmentation techniques. We used the ImageDataGenerator object in Keras to perform our augmentation. The parameters we selected for augmentation were rotation\_range=180 degrees, width\_shift\_range = 0.1, height\_shift\_range=0.1, shear\_range=0.1, zoom\_range=0.01, and fill\_mode='nearest'. The image set was split into training and validation sets. The training images set was then used for augmentation. Since the masks had to be augmented with training images, we created two ImageDataGenerator objects, one for the training images and one for the training masks. We kept the augmentation the same by passing in the same seed parameter value in the 'flow()' function for our generators. We augmented the data before each epoch using the 'fit_generator()' function for our model.
- show example of augmented images with masks

### Implementation
Code implementation [Add code]:
- data loading class (ImageData)
- UNet Convolutional Neural Network model (UNet) [Add model architecture image with dimensions]
- submission code (testevaluation.py)

Testing implementation:
1. Load weights, and predict the mask for each testing image.
2. Create contour for each mask by using OpenCV 'findContours()' function. 
3. Output [x,y] points to properly labeled text file using eval python class.
4. Send contour files to moderators for Dice coefficient on Test set.

The model implementation involved three separate phases:
1. Data preprocessing and loading
2. Model development and training
3. Test data evaluation and creation of submission files.

In order to load the data into memory and serve it as input for training the model, we had to create a custom Python class named ImageData. The ImageData class loads the patient images from the DICOM files, creates binary masks from the contours provided, and loads the masks into memory as well.  To load images from the DICOM files provided, we used the pydicom library. The library has a read\_file function that will load the MRI image, and then we store it as a numpy array. The contours were provided as text files with (x,y) coordinates corresponding to the pixel that was part of the right ventricle region of interest. To create these masks from the contours, we used the Pillow library to create a new Image object and create a polygon with the same outline as the contour points given. The area of the shape defined by that polygon would contain white pixels while the pixels outside that region would be black. This is how we created the binary masks from the contours given. We stored these masks in a list which was accessed from an instance of the ImageData class. In addition to loading the images and masks, we had to rotate some of the images. Most of the images were 256x216, however there were a few patients that had images that were 216x256. In those cases, we had to rotate the images to be 256x216 in size. In addition to rotating the images, we had to rotate our binary masks as well. 

After creating the data loading class, we had to build our convolutional neural network. The neural network architecture we followed was the U-Net structure proposed by Rottenberger et al. The U-Net model is made of a downsampling block followed by an upsampling block to create the segmentation map of similar dimensions as the input image. The downsampling block is a series of 2 convolutional layers followed by a max-pooling layer. The convolutional layers use the ReLU activation function, 3x3 filter size, 32 features at the start, and padding='same'. The max-pooling layer had a pool size of 2. At each downsampling step, the feature size was doubled, and this block of conv=>conv=>max\_pool was repeated for 3 steps. At each step, we were also making copies of the output from the second convolution to be used for our upsampling block. When upsampling, we had an up-convolution step followed by two convolutions. At each step we were halving the number of filters for our layers. The output of the model was a binary segmentation mask of similar dimensions as the input. This was created by applying one last convolution with filter size 2, kernel size 1, and a softmax activation function. The output mask labeled a white pixel as part of the right ventricle, while a black pixel was part of the background.

We implemented the U-Net CNN using the deep learning library, Keras. We built our network on a Tensorflow backend. The downsampling block used the standard Conv2D object for our convolutional layers. We used a MaxPooling2D layer for our max-pooling layer. For the upsampling block of the network, we had to perform an transposed convolution followed by two Conv2D layers. To perform the transposed convolution, we used the ConvTranspose2D object from Keras. The code for our implementation can be found in Figure XX. 

After creating the ImageData class and the UNet class, we were ready to train our model. All the training was done within a Jupyter Notebook. In order to train the Unet model, we followed the following steps: 
Training implementation:
1. Load images and masks into memory : This was done as described using the ImageData class.
2. Split images and masks into training and validation sets
   - We split the training set into 80% training data and 20% validation. 
3. Augment training data as described above using ImageDataGenerator object.
4. Load the model architecture.
5. Train the model using 'fit\_generator()' and train image and masks generators as input data. Validation images and masks as validation data.
6. Calculate the average Training and Val Dice by predicting the trained model on training images and validation images using 'predict()' function.


### Refinement
- Change dropout
- change numbe of epochs
- batch normalization
- loss function; learning rate.
## IV. Results
### Model Evaluation and Validation
### Justification

## V. Conclusion
### Free-Form Visualization
### Reflection
### Improvement
