# Computer Vision Project 
## Members: 
Adrian Gheorghiu

Nazanin Niayesh

## Abstract
In this project, we will train a classifier model to assign an emotion such as anger or happiness to a given expression using the [Face expression recognition dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)[[1]](#1). 
This dataset includes grayscale pictures of 7 different expression categories, namely: anger, disgust, fear, sadness, happy, neutral and surprise. 
We will then optimize this model for mobile and develop an android mobile app that uses this model to recognize facial expressions based on photos taken on the phone and/or real-time camera feed of the device.

## Introduction
Facial Expression Recognition (FER) is a task in Computer Vision which aims to automate the process of identifying and classifying emotional expressions of human faces. Different algorithms often use the form and position of facial features such as mouth, eyes or eyebrows amongst others to determine what emotion the facial expression depicted in an image or a video corresponds to. The importance and range of applications of FER has increased with recent technological advances and improved capability of cameras resulting in better quality of images and videos that can be used to train FER models. Application areas of FER include but are not limited to personalized services (e.g. providing personalized messages depending on the user’s current emotional state in a smart environment), healthcare (e.g. detecting early signs of depression, monitoring patient state during treatment) and education (e.g. detecting engagement and emotional reaction of students or monitoring student attention) [[2]](#2).

## Problem Description
The aforementioned FER models are often Neural Networks that are trained on images or videos of human faces with different expressions, and then tested and used to categorize emotions given a new set of such images or videos. The training data are often divided into several different categories of emotions (of different complexities). In this project, we will be using the “Face Expression Recognition Dataset” [[1]](#1) which includes images of 7 fundamental expression categories, namely: Angry, disgusted, fearful, sad, happy, neutral and surprised. The bar plot below shows the number of available data for each category of the dataset. It is noticeable that the number of data present especially for the class “disgusted” is significantly lower than that of other classes. This will be discussed and handled further in the “Experiments” section.

[IMAGE]

The aim of this project is to solve the problem of accurately and efficiently classifying any given facial expression into one of the categories mentioned above in real-time. To achieve this, we will train a classifier model on the training data provided by the aforementioned Face Expression Recognition dataset. After training, this model should be capable of successful classification of a given facial expression into one of the corresponding emotion categories without being (negatively) affected by factors such as the quality and resolution of the given image, the lighting, the person’s background and skin color/ethnicity, etc. 

## Proposed solution
To achieve the goal of accurate and efficient real-time facial expressions recognition, we will fine tune pre-trained classifier models on the training data provided by the aforementioned Face Expression Recognition dataset. Additionally, we will develop a mobile android app which facilitates the use of the developed model for (real-time) classification of given facial expressions and allows more user-friendly access to our work for a wider audience (incl. users with less or non-existing technical backgrounds). Some possible challenges to be aware of while developing the app include the limitations it imposes on the size of the FER models (models that are too large might reduce app performance and speed, too small might not be accurate enough), designing a simple and user-friendly interface that at the same time includes all desired and necessary functionalities, and ensuring smooth function of the real-time detection feature.

## Experiments
To address the data imbalance regarding the "disgusted" category of the training data from the "Face Expression Recognition" dataset, we decided to create additional synthetic images of faces depicting disgusted expression. For this purpose, we train a StarGan [[3]](#3) generative adversarial network to translate between the different emotions in the dataset, with the ultimate goal of generating synthetic "disgusted" images by translating images of other emotions. Some examples of these generated images can be seen below where the image on the left is the original and the image on the right is the translated image depicting a disgusted face. 

[IMAGES]

To balance the dataset, we chose to augment the "disgusted" class until its number of samples is the same as the "fear class". The sample distribution after adding the new synthetic images is depicted in the bar plot below. As it can be seen, it is now more balanced regarding the number of data for each of the 7 emotion categories. We will test both the augmented and unaugmented datasets with the chosen models to ascertain the improvement augmentation brings to the table.

[IMAGE]

To find an appropriate model which fulfills the requirements for the proposed solution, we experimented with two different Vision Transformer (ViT) models which process images using self-attention mechanisms to capture dependencies and relations between image patches. This self-attention mechanism allows ViT to capture global dependencies better compared to e.g. Convolutional Neural Networks (CNNs) that extract features hierarchically using their convolutional layers. Such global dependencies may be of importance in determining correct facial expression as is the goal of this project and therefore we decided to train and use two different sized ViT models (ViTtiny and ViTsmall) to compare their performance (also on the mobile app). These models achieve faster inference speeds and a smaller memory footprint compared to the base ViT by compromising on the number of attention layers, the hidden embedding size, and the number of attention heads respectively. Other models like "CoaT-Lite tiny" and "MobileNet v2" were also trained but without any extensive testing. Additionally, we tested different augmentation functions for the developed FER model (used for online on-the-fly image augmentation at train time). We tested the following functions for both ViT models trained on the augmented dataset: 
* Cutout: an image augmentation strategy that randomly masks out square regions of input during training (object occlusion).
* Cutmix: instead of simply removing pixels from the image like cutout, the removed regions are replaced with patches from another image.
* Fmix: similar to cutmix but uses masks sampled from fourier space to mix 2 training samples.
* Mixup: linearly combines 2 training samples by creating a weighted combination of them.
The result of these combinations are shown and discussed in the following “Results” section. 
 
In order to increase the robustness of the developed solution, we implemented some out-of-distribution (OOD) detection. OOD detection refers to the ability of a trained model to recognize and handle data which deviate from its training set. We used a distance-based approach, specifically relative Mahalanobis distance, to determine which samples deviate significantly from the training data. The relative Mahalanobis distance between a given data (point) z’ and a distribution for a certain class is formally defined as: $RMD_k(z')=MD_k(z')-MD_0(z')$ , where $MD_k$ is the Mahalanobis distance to the distribution of class k, and $MD_0$ is the distance to the distribution of the whole dataset. The Mahalanobis distance is defined as: $MD_k(z')=(z'-\mu_k)^T\Sigma^{-1}(z'-\mu_k)$, where $\mu_k$ is the mean vector of class k and $\Sigma$ is the covariance matrix.

## Results (quantitative)
In this section, the results of the aforementioned ViT models, ViTtiny and ViTsmall with and without variations in the online augmentation function using the augmented or unaugmented training datasets, are shown.

ViTtiny:
|         | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---------|----------|-----------------|--------------|----------|
| Cutout  | 0.667    | 0.656           | 0.654        | 0.639    |
| Mixup   | 0.688    | 0.684           | 0.664        | 0.667    |
| Cutmix  | 0.688    | 0.688           | 0.669        | 0.670    |
| Fmix    | 0.695    | 0.694           | 0.681        | 0.677    |
| Fullaug | 0.670    | 0.675           | 0.649        | 0.650    |
| Unaug   | 0.665    | 0.644           | 0.645        | 0.630    |

ViTsmall:
|         | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---------|----------|-----------------|--------------|----------|
| Cutout  | 0.666    | 0.670           | 0.641        | 0.651    |
| Mixup   | 0.672    | 0.677           | 0.648        | 0.660    |
| Cutmix  | 0.667    | 0.656           | 0.647        | 0.650    |
| Fmix    | 0.688    | 0.668           | 0.672        | 0.670    |
| Fullaug | 0.686    | 0.686           | 0.662        | 0.672    |
| Unaug   | 0.686    | 0.689           | 0.654        | 0.667    |

## Results (Qualitative)
Additionally, we rendered the Class Activation Maps (CAM) for two highlighted images in order to visualize what models trained using different online activations learn to focus their attention on. 

[IMAGES]

The images on the left at each of these figures give a practical example for the method definition as described in the “Experiments”' section. On the right of each figure, the attention of the model can be seen to be directed to different facial features such as the eyes (e.g. in cutmix and fmix), the eyebrows and lips (e.g. in mixup and cutmix). 

[IMAGE]

Above we can see pictures from the app. The user has the possibility to turn real time detection on or off. When on, the app continuously does OOD detection and inference and displays the result below the image. When real time detection is off, the user has the possibility to capture an image and once OOD detection is run, they have the possibility of detecting the emotion. There are currently 3 models the user can choose from: ViT-small, ViT-tiny and MobileNet-V2. For OOD detection, the MobileNet-V2 features are used.

## Conclusion
As it can be seen above, ViTtiny model with the Fmix augmentation function has the highest overall accuracy of 0.695 and highest Macro F1 score (average F1 score across all emotion classes) of 0.677. Similarly, the ViTsmall model with the same Fmix augmentation function performs the best compared to other ViTsmall variations with an accuracy of 0.688 and Macro F1 score of 0.670 so it can be concluded that the Fmix augmentation functions works best for this facial expression recognition task. An explanation for this may be that this way the model learns to focus on what actually makes a face have a certain emotion and helps it not overfit on specific traits, while at the same time providing a more natural and better spectrally responsive way of mixing the two images.

Additionally, ViTtiny performs slightly better than the ViTsmall model with the mixup, CutMix and FMix augmentation functions while ViTsmall significantly outperforms the ViTtiny model with the cutout (significantly higher Macro F1 score) augmentation method and on both the augmented and unaugmented dataset. This suggests that the improvements of the larger VitSmall model are not only minimal but also encourage overfitting. The effects of the dataset augmentation are also reflected in the accuracy and macro F1 scores of both models. It can be seen that the accuracy and macro F1 score of ViTtiny and the macro F1 score ViTsmall model are higher when trained on the augmented dataset.  

Regarding the developed android mobile app, the ViTtiny model works smoother and is sometimes quicker to respond due to its smaller size. On the other hand, the ViTsmall model also performs relatively well and delivers more accurate emotion recognition results of a given face which is justified by the results above. The real-time feature works seamlessly in the app and can be turned on or off depending on the user’s preference.

## Future Work
This project experimented with a few classifier models for facial expressions with variations in training datasets and image augmentation strategies. Future work could explore the performance of other models such as CNNs (which were mentioned as alternatives in the previous sections) or other Transformer-based architectures besides the two models discussed here. It is possible to experiment with adjusting the number of the synthetic images used for augmenting the “disgusted” category of the dataset, and to possibly augment other classes of the dataset besides “disgusted” or use a different dataset of facial expressions altogether. Additionally, other online image augmentation strategies such as color-space-augmentation methods (adjusting brightness, contrast, saturation, etc.) could be implemented which might positively affect the performance of the model.

## References
<a id="1">[1]</a> 
Jonathan Oheix.
Face Expression Recognition Dataset <https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset>.
<a id="2">[2]</a> 
European Data Protection Supervisor (EDPS) TechDispatch. 
Facial Emotions Recogntion, Issue 1 (2021), P.2.
<a id="3">[3]</a> 
Yunjey Choi et al.
StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation.
2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), P.8789-8797.
doi: 10.1109/CVPR.2018.00916.








