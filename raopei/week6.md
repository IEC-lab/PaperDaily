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

小目标检测一直是目标检测中的难点，主要由于小目标面积过小，在RPN的Anchor内，容易通过IoU设置将其丢弃，还会存在CNN提取的高层语义特征容易与分辨率产生矛盾，致使检测的效果极差。为了解决这个问题，主要思路就是在训练和反向传播更新参数时，只考虑哪些在指定的尺度范围内的目标，由此提出了一种特别的多尺度训练方法，即SNIP（Scale Normalization for Image Pyramids）。\
在训练时，划分了三个尺度，对应三种不同分辨率的图像。每个分辨率下的RoI都有其指定范围，如果GT的bounding-box大小在这个范围内，就被标记做valid，否则就被标记为invalid。
这种做法，最终的检测效果大大提升，但是训练过程比较复杂：

作者使用的是Deformable RFCN detector而不是常见的一般卷积；

作者使用的网络结构是Dual path networks（DPN）和ResNet-101，由于需要内存很大，为了适应GPU内存，作者对图像进行了采样，具体方法是选取一个1000x1000的包含最多目标的区域作为子图像，然后重复该步骤直到所有目标都被选取 ；

作者为了提升RPN的效果，尝试了使用7个尺度，连接conv4和conv5的输出。
