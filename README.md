# Cataract-Detection-and-Classification

![image](https://user-images.githubusercontent.com/48744487/119627845-2e529500-be2a-11eb-9c19-9967cc5e7d4f.png)

Our system works on the detection of cataracts and type of classification on the basis of severity namely; mild, normal, and severe, in an attempt to reduce errors of manual detection of cataracts in the early ages.
The phase 1 implementation has successfully classified images as cataract affected or as a normal eye with an accuracy of 96% using combined feature vectors from the SIFT-GLCM algorithm applied to classifier models of SVM, Random Forest, and Logistic Regression. The effect of using SIFT and GLCM separately has also been studied which leads to comparatively lesser accuracies in the model trained. 
The phase 2 implementation which deals with the type classification, has obtained the maximum validation acurracy of 97.66% using deep convolutional neural network models, in particular SqueezeNet, MobileNet, and VGG16.
The results have been made accessible using web and Flask based user interface.
The phase 1 implementation of the project which works on binary classification of cataract has been compiled into a conference paper and accepted in the “International Conference on Artificial Intelligence: Advances and Applications (ICAIAA 2021).”
Algorithms used

PHASE 1

1. SIFT 
2. GLCM
3. SVM
4. LOGISTIC REGRESSION
5. RANDOM FOREST
6. KNN

PHASE 2
1. HOUGH CIRCLE TRANSFORM
2. VGG-16
3. MOBILENET V2
4. SQUEEZENET
