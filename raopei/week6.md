# Paper：An Analysis of Scale Invariance in Object Detection – SNIP

## Abstract

An analysis of different techniques for recognizing and
detecting objects under extreme scale variation is presented. Scale specific and scale invariant design of detectors are compared by training them with different configurations of input data. By evaluating the performance
of different network architectures for classifying small objects on ImageNet, we show that CNNs are not robust to
changes in scale. Based on this analysis, we propose to
train and test detectors on the same scales of an imagepyramid. Since small and large objects are difficult to recognize at smaller and larger scales respectively, we present
a novel training scheme called Scale Normalization for Image Pyramids (SNIP) which selectively back-propagates the
gradients of object instances of different sizes as a function
of the image scale. On the COCO dataset, our single model
performance is 45.7% and an ensemble of 3 networks obtains an mAP of 48.3%. We use off-the-shelf ImageNet-1000
pre-trained models and only train with bounding box supervision. Our submission won the Best Student Entry in
the COCO 2017 challenge. Code will be made available at
http://bit.ly/2yXVg4c.

### Contribution

小目标检测一直是目标检测中的难点，主要由于小目标面积过小，在RPN的Anchor内，容易通过IoU设置将其丢弃，还会存在CNN提取的高层语义特征容易与分辨率产生矛盾，致使检测的效果极差。
