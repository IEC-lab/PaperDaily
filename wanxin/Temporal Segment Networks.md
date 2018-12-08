# Temporal Segment Networks: Towards Good Practices for Deep Action Recognition

ECCV2016，[PDF](https://arxiv.org/abs/1608.00859v1) ，[Models and code](https://github.com/yjxiong/temporal-segment-networks)

> **Abstract.**Deep convolutional networks have achieved great success for visual recognition in still images. However, for action recognition in videos, the advantage over traditional methods is not so evident. This paper aims to discover the principles to design effective ConvNet architectures for action recognition in videos and learn these models given limited training samples. Our first contribution is temporal segment network (TSN), a novel framework for video-based action recognition. which is based on the idea of long-range temporal structure modeling. It combines a sparse temporal sampling strategy and video-level supervision to enable efficient and effective learning using the whole action video. The other contribution is our study on a series of good practices in learning ConvNets on video data with the help of temporal segment network. Our approach obtains the state-of-the-art performance on the datasets of HMDB51 (69:4%) and UCF101 (94:2%). We also visualize the learned ConvNet models, which qualitatively demonstrates the effectiveness of temporal segment network and the proposed good practices.

**摘要：** 本文旨在设计有效的卷积网络体系结构用于视频中的动作识别，并在有限的训练样本下进行模型学习。TSN基于two-stream方法构建。论文主要贡献：

* 提出了TSN（Temporal Segment Networks），基于长范围时间结构（long-range temporal structure）建模，结合了稀疏时间采样策略（sparse temporal sampling strategy）和视频级监督（video-level supervision）来保证使用整段视频时学习得有效和高效
* 在TSN的帮助下，研究了一系列关于视频数据学习卷积网络的良好实践

数据集表现：HMDB51(69.4%)、UCF101（94.2%）

## Introduction

作者认为，基于视频的动作识别受两个主要因素的阻碍：

* 首先，在很多短视频特别是运动视频中，它的动作都是经历了比较长的时间，而主流卷积网络框架通常关注于单帧或短时间的动作。目前主要解决方法是密集时间采样，这种方法在应用于长视频序列时会产生过多的计算成本，限制了它在实际应用中的应用，并且对于超过最大序列长度的视频，存在丢失重要信息的风险。
* 其次，训练深层神经网络需要大量的数据，但因为动作数据难收集和注释，使得现有的数据集在大小和多样性方面仍然有限

这些挑战促使我们研究两个问题：

* 如何设计一种有效的基于视频的网络结构能够学习视频的表现进而捕捉 long-range 时间结构
* 如何在有限的训练样本下学习卷积神经网络模型

针对第一个问题，在时间结构建模方面，一个关键的观察结果是连续的帧是高度冗余的。因此，密集时间采样是不必要的，它通常会产生高度相似的采样帧。相反，稀疏时间采样策略在这种情况下会更有利。基于这一观察，作者开发了Temporal Segment Networks，该框架采用稀疏采样的方法提取长视频序列中的短片段，这种策略以极低的成本保留了相关信息，从而在合理的时间和计算资源预算下实现了长视频序列的端到端学习

针对第二个问题，主要通过以下三个方法来解决：(1) cross-modality pre-training；(2) regularization；(3) enhanced data augmentation

## Network Architecture

![1](image\TSN\1.png)

对于一个输入的视频，被分成K个段（segment），实验中设置段的数量为 3。从每个段中随机地选择一个片段（short snippet）。将选择的片段通过two-stream卷积神经网络得到不同片段的类别分数，最后将它们融合。不同片段的类别得分采用段共识函数（The segmental consensus function）进行融合来产生段共识（segmental consensus），这是一个视频级别的预测。然后对所有模式的预测使用标准分类交叉熵损失（standard cross-entropy loss）融合产生最终的预测结果（这种方法大大降低了计算开销），采用随机梯度下降法(SGD)训练网络

网络对片段序列的建模如下，Tk表示随机选择的片段，F函数表示卷积网络，G为聚合函数，H为softmax函数

![2](image\TSN\2.png)

**网络的损失函数**如下（standard categorical cross-entropy），C表示类别数，yi是ground truth

![3](image\TSN\3.png)

在反向传播过程中梯度如下

![4](image\TSN\4.png)

### 网络细节

1. 网络选择Inception with Batch Normalization (BN-Inception)

2. two-stream卷积神经网络只将RGB图像和光流分别作为时间和空间流的输入。为了增强网络的泛化能力，作者在原来基础上提出增加帧差图像和warped光流，四种模式图像如下

   ![5](image\TSN\5.png)

### 网络训练

由于行为检测的数据集相对较小，训练时有过拟合的风险，为了缓解这个问题，作者设计了几个训练策略

#### cross-modality pre-training

空间网络以RGB图像作为输入：故采用在ImageNet上预训练的模型做初始化。对于其他输入模式（比如：帧差图像和光流场），它们基本上捕捉视频数据的不同视觉方面，并且它们的分布不同于RGB图像的分布。作者提出了交叉模式预训练技术：利用RGB模型初始化时间网络

首先，通过线性变换将光流场离散到从0到255的区间，这使得光流场的范围和RGB图像相同。然后，修改RGB模型第一个卷积层的权重来处理光流场的输入。具体来说，就是对RGB通道上的权重进行平均，并根据时间网络输入的通道数量复制这个平均值。这一策略对时间网络中降低过拟合非常有效

#### regularization

在学习过程中，Batch Normalization将估计每个batch内的激活均值和方差，并使用它们将这些激活值转换为标准高斯分布。这一操作虽可以加快训练的收敛速度，但由于要从有限数量的训练样本中对激活分布的偏移量进行估计，也会导致过拟合问题。因此，在用预训练模型初始化后，冻结所有Batch Normalization层的均值和方差参数，但第一个标准化层除外。由于光流的分布和RGB图像的分布不同，第一个卷积层的激活值将有不同的分布，于是，我们需要重新估计的均值和方差，称这种策略为部分BN。与此同时，在BN-Inception的全局pooling层后添加一个额外的dropout层，来进一步降低过拟合的影响。dropout比例设置：空间流卷积网络设置为0.8，时间流卷积网络设置为0.7

#### enhanced data augmentation

数据增强能产生不同的训练样本并且可以防止严重的过拟合。在传统的two-stream中，采用随机裁剪和水平翻转方法增加训练样本。作者采用两个新方法：角裁剪（corner cropping）和尺度抖动（scale-jittering）

**角裁剪（corner cropping）**：仅从图片的边角或中心提取区域，来避免默认关注图片的中心

**尺度抖动（scale jittering）**：将输入图像或者光流场的大小固定为 256×340256×340，裁剪区域的宽和高随机从 {256,224,192,168}{256,224,192,168} 中选择。最终，这些裁剪区域将会被resize到 224×224224×224 用于网络训练。事实上，这种方法不光包括了尺度抖动，还包括了宽高比抖动

## Experiments

作者对四种方案进行实验：（1）从零开始训练；（2）仅仅预训练空间流；（3）采用交叉输入模式预训练；（4）交叉输入模式预训练和部分BN dropout结合。结果总结在下表1中

![6](image\TSN\6.png)

由上表可以看出，从零开始训练比基线算法（two-stream卷积网络）的表现要差很多，证明需要重新设计训练策略来降低过拟合的风险，特别是针对空间网络。对空间网络进行预训练、对时间网络进行交叉输入模式预训练，取得了比基线算法更好的效果。之后还在训练过程中采用部分BN dropout的方法，将识别准确率提高到了92.0%



在上文中提出了两种新的模式：RGB差异和扭曲的光流场。不同输入模式的表现比较如下表2中

![7](image\TSN\7.png)

由上表可以看出，首先，RGB图像和RGB差异的结合可以将识别准确率提高到87.3%，这表明两者的结合可以编码一些补充信息。光流和扭曲光流的表现相近（87.2% vs 86.9%），两者融合可以提高到87.8%。四种模式的结合可以提高到91.7%。由于RGB差异可以描述相似但不稳定的动作模式，作者还评估了其他三种模式结合的表现（92.3% vs 91.7%）。作者推测光流可以更好地捕捉运动信息，而RGB差异在描述运动时是不稳定的。在另一方面，RGB差异可以当作运动表征的低质量、高速的替代方案



段共识函数被定义为它的聚合函数 g，这里评估 g 的三种形式：（1）最大池化；（2）平均池化；（3）加权平均。实验结果见表3中

![8](image\TSN\8.png)

我们发现平局池化函数达到最佳的性能。因此在接下来的实验中选择平均池化作为默认的聚合函数。然后比较了不同网络架构的表现，结果总结在表4

![9](image\TSN\9.png)

具体来说，比较了3个非常深的网络架构：BN-Inception、GoogLeNet和VGGNet-16。在这些架构中，BN-Inception表现最好，故选择它作为TSN的卷积网络架构



与其他视频行为识别算法在HMDB51和UCF101数据集上的对比，效果还是比较明显的，结果如下

![10](image\TSN\10.png)

## Conclusion

本文对two-stream系列网络进行了优化，主要解决了在长时间视频预测和数据集受限两个问题。这项工作在保持合理的计算成本的同时将准确率提升到了state-of-the-art，在动作识别这一领域属于比较有代表性的一篇文章
