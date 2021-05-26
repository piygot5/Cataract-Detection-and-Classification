# Cataract-Detection-and-Classification

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/48744487/119628159-77a2e480-be2a-11eb-8557-eb8186d6fe04.png">
</p>

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

If you wish to learn before in-depth about GLCM texture feature extraction algorithm you refer the following written by [Kamaljit Kaur](https://github.com/kamaljitkaur98)

https://www.notion.so/Understanding-GLCM-7d2501afd206430b906e4a9851e86280
