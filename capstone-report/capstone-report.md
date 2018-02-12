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



