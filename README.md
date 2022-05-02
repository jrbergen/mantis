# tmap_defectdetector

ML based defect detector (Project in planning phase).

Main steps involved in the development of a Machine Learning Algorithm are:

Import dataset (Input and Output)
          |
          \/
Pre-processing (Filtering, upsizing/downsizing)
          |
          \/
Build model or fit regressions to data
          |
          \/
Vizualize Results Graphically and 
compare with validation

## Part I - Wine Quality Predictor
- Here we see a linear regression Machine Learning Algorithm applied to a generalized data for 
determining quality of wine.
- The dataset provides different parameters as input for determining quality such as alcohol level, pH, chlorides, etc. However for this project, volatile acidity was chosen as it shows a correlation with quality. See image below:

![Alt text](/Images/Raw%20Data%20Quality%20vs%20Volatile%20acidity.png?raw=true "Title")

- Final Linear Regression curve was found to be with slope -1.7 and intercept 6.5.

![Alt text](/Images/Final%20Plot.png?raw=true "Title")

No interface was developed for this part of the project as it was not necessary for the basic linear regression.

## Part II - Image Defect Predictor
Here, a machine learning algorithm which utilizes a Convolutional Neural Network is used to detect and classify 
defects on solar panels based on surface scratches on them.
-
-
.
.
.
