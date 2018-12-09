# Paper:Going deeper with convolutions

## Abstract

We propose a deep convolutional neural network architecture codenamed Inception,
which was responsible for setting the new state of the art for classification
and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014
(ILSVRC14). The main hallmark of this architecture is the improved utilization
of the computing resources inside the network. This was achieved by a carefully
crafted design that allows for increasing the depth and width of the network while
keeping the computational budget constant. To optimize quality, the architectural
decisions were based on the Hebbian principle and the intuition of multi-scale
processing. One particular incarnation used in our submission for ILSVRC14 is
called GoogLeNet, a 22 layers deep network, the quality of which is assessed in
the context of classification and detection

## Contribution

GoogleNet的开山之作，提出了获得高质量模型最保险的做法就是增加模型的深度（层数）或者是其宽度，并在实践中设计出了这个网络，
解决了增加深度、宽度存在的一些问题。\
1.深度，层数更深，文章采用了22层，为了避免梯度消失问题，googlenet巧妙的在不同深度处增加了两个loss来保证梯度回传消失的现象。\
2.宽度，增加了多种核 1x1，3x3，5x5，还有直接max pooling的，但是如果简单的将这些应用到feature map上的话，concat起来的feature map
厚度将会很大，所以在googlenet中为了避免这一现象提出的inception具有如下结构，在3x3前，5x5前，max pooling后分别加上了1x1的卷积核起
到了降低feature map厚度的作用。

# Paper:Deep Residual Learning for Image Recognition

## Abstract
Deeper neural networks are more difficult to train. We
present a residual learning framework to ease the training
of networks that are substantially deeper than those used
previously. We explicitly reformulate the layers as learning
residual functions with reference to the layer inputs, instead
of learning unreferenced functions. We provide comprehensive
empirical evidence showing that these residual
networks are easier to optimize, and can gain accuracy from
considerably increased depth. On the ImageNet dataset we
evaluate residual nets with a depth of up to 152 layers—8×
deeper than VGG nets [41] but still having lower complexity.
An ensemble of these residual nets achieves 3.57% error
on the ImageNet test set. This result won the 1st place on the
ILSVRC 2015 classification task. We also present analysis
on CIFAR-10 with 100 and 1000 layers.
The depth of representations is of central importance
for many visual recognition tasks. Solely due to our extremely
deep representations, we obtain a 28% relative improvement
on the COCO object detection dataset. Deep
residual nets are foundations of our submissions to ILSVRC
& COCO 2015 competitions1
, where we also won the 1st
places on the tasks of ImageNet detection, ImageNet localization,
COCO detection, and COCO segmentation.

## Contribution
随着网络的加深，出现了训练集准确率下降的现象,并且人们确定这不是由于过拟合造成的，
为了解决这个问题，作者提出了这个网络。对于“随着网络加深，准确率下降”的问题，Resnet提供了两种选择方式，
也就是identity mapping和residual mapping，如果网络已经到达最优，继续加深网络，residual mapping将被push为0，
只剩下identity mapping，这样理论上网络一直处于最优状态了，网络的性能也就不会随着深度增加而降低了。
