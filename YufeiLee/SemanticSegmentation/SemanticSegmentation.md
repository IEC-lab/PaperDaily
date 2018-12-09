# FCN

**paper:**[Fully Convolutional Networks for Semantic Segmentation](<https://arxiv.org/abs/1411.4038>)`CVPR (2015)`

## Abstract

卷积网络在特征分层领域是非常强大的视觉模型。我们证明了经过端到端、像素到像素训练的卷积网络超过语义分割中最先进的技术。我们的核心观点是建立“全卷积”网络，输入任意尺寸，经过有效的推理和学习产生相应尺寸的输出。我们定义并指定全卷积网络的空间，解释它们在空间范围内dense prediction任务(预测每个像素所属的类别)和获取与先验模型联系的应用。我们改编当前的分类网络(AlexNet ,the VGG net, and GoogLeNet)到完全卷积网络和通过微调传递它们的学习表现到分割任务中。然后我们定义了一个跳跃式的架构，结合来自深、粗层的语义信息和来自浅、细层的表征信息来产生准确和精细的分割。我们的完全卷积网络成为了在PASCAL VOC最出色的分割方式（在2012年相对62.2%的平均IU提高了20%），NYUDv2，和SIFT Flow,对一个典型图像推理只需要花费不到0.2秒的时间。

## Contributions



# Attention to Scale

**paper:**[Attention to Scale: Scale-aware Semantic Image Segmentation](<https://arxiv.org/abs/1511.03339> )`CVPR2016`

## Abstract

在完全卷积神经网络（FCN）中结合多尺度特征一直是实现语义图像分割的最先进性能的关键因素。提取多尺度特征的一种常用方法是将多个已调整大小的输入图像馈送到共享深度网络，然后合并所得到的特征以进行按像素分类。在这项工作中，我们提出了一种注意机制，可以学习对每个像素位置的多尺度特征进行轻微加权。我们采用最先进的语义图像分割模型，与多尺度输入图像和注意模型共同训练。所提出的注意模型不仅优于平均和最大池化，而且允许我们诊断地可视化在不同位置和尺度上特征的重要性。此外，我们表明，在合并多尺度特征时，为每个尺度的输出添加额外的监督对于实现卓越的性能至关重要。我们通过对三个具有挑战性的数据集进行了大量实验来证明我们的模型的有效性，包括PASCAL-Person-Part，PASCAL VOC 2012和MS-COCO 2014的子集。

## Contributions

 将注意力机制用到多分辨率输入的语义分割网络中。 

1.首先，多尺度的特征是提升语义图像分割性能的一个关键因素。
2.提取多尺度的特征主要有两种网络结构：
  第一种是：skip-net，第二种是：share-net

(1)skip-net的特点是取网络中多个中间层的特征并合并成一个特征，以实现多尺度的特征；
(2)share-net的特点是对输入图像进行尺度上的变换，得到不同尺度的输入图像，然后分别输入给网络，这样能够得到不同尺度的输入图像的特征，以形成多尺度的特征。
3.论文采用的是share-net的方式来得到多尺度的特征，在采用share-net方式提取多尺度特征时，需要考虑到一个问题，就是如何对多个尺度输入图像得到的特征进行融合？
(1)多尺度输入图像的特征的融合目前主要有两种方式，一种是max pooling；一种是average pooling(取平均)；

(2)本篇论文提出对多尺度输入图像特征进行加权求和进行融合

#  DeepLab

**paper:**[DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](<https://arxiv.org/abs/1606.00915> )`TPAMI`

## Abstract

DCNNs近期在高级视觉任务中表现出非常好的性能，比如图像分类和目标跟踪。本文联合DCNNs和概率图模型来解决像素级分类任务,也就是语义图像分割。我们发现DCNNs最后一层的响应不能充分地用于定位精确的目标分割。这是因为存在使得DCNNs擅长高级任务的非常的不变性属性。我们通过一个全连接的CRF结合DCNNs最后层的响应来克服深度网络中差的定位属性。定性的讲，我们的”DeepLab”系统能够以超过先前方法的精度来定位分割边界。定量的讲，我们的方法在PASCAL VOC2012语义图像分割任务上测试的平均IOU达到71.6%。我们展示了如何能有效地得到这些结果:注意网络的再利用；从小波社区引入”孔洞”算法，该算法在现代GPU上能以每秒8帧的速度处理神经网络响应的密集计算.

## Contributions

(1)速度: 由于“atrous”算法和感受野尺寸的减小，我们的密集DCNN以8 fps的速度运行，而完全连接的CRF的平均场推理需要0.5秒 
(2)精度: PASCAL语义分割挑战最现先进结果，超过了Mostajabi等人的方法的2％ 
(3)简单性: 我们的系统由两个相当完善的DCNN和CRF模块组成。



# DeepLab v3

**paper:**[Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](<https://arxiv.org/abs/1802.02611> )`ECCV2018`

## Abstract

空间金字塔池模块或编码 - 解码器结构用于深度神经网络中解决语义分割任务。前一种网络能够通过利用多个速率和多个有效视场的过滤器或池化操作探测输入特征来编码多尺度上下文信息，而后一种网络可以通过逐渐恢复空间信息来捕获更清晰的对象边界。在这项工作中，我们建议结合两种方法的优点。具体来说，我们提出的模型DeepLabv3 +通过添加一个简单而有效的解码器模块来扩展DeepLabv3，以优化分割结果，尤其是沿着对象边界。我们进一步探索Xception模型并将深度可分离卷积应用于Atrous Spatial Pyramid
Pooling和解码器模块，从而产生更快更强的编码器-解码器网络。我们证明了所提出的模型在PASCAL VOC 2012语义图像分割数据集上的有效性，并且在没有任何后处理的情况下在测试集上实现了89％的性能。我们的论文附有Tensorflow中提出的模型的公开参考实现。

## Contributions

•我们提出了一种新颖的编码器-解码器结构，它采用DeepLabv3作为功能强大的编码器模块和简单而有效的解码器模块。

•在我们提出的编码器 - 解码器结构中，可以通过atrous卷积任意控制提取的编码器特征的分辨率，以折中精度和运行时间，这对于现有的编码器解码器模型是不可能的。

•我们将Xception模型用于分割任务，并将深度可分离卷积应用于ASPP模块和解码器模块，从而产生更快更强的编码器-解码器网络。

•我们提出的模型在PASCAL VOC 2012数据集上获得了新的最新性能。我们还提供设计选择和模型变体的详细分析。

•我们公开提供基于Tensorflow的提议模型实现。
