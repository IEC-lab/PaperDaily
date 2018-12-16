# FCN

**paper:**[Fully Convolutional Networks for Semantic Segmentation](<https://arxiv.org/abs/1411.4038>)`CVPR (2015)`

**code：**[fcn](https://github.com/shelhamer/fcn.berkeleyvision.org)

**translate**：[fcn](https://www.jianshu.com/p/91c5db272725)

## Abstract

卷积网络在特征分层领域是非常强大的视觉模型。我们证明了经过端到端、像素到像素训练的卷积网络超过语义分割中最先进的技术。我们的核心观点是建立“全卷积”网络，输入任意尺寸，经过有效的推理和学习产生相应尺寸的输出。我们定义并指定全卷积网络的空间，解释它们在空间范围内dense prediction任务(预测每个像素所属的类别)和获取与先验模型联系的应用。我们改编当前的分类网络(AlexNet ,the VGG net, and GoogLeNet)到完全卷积网络和通过微调传递它们的学习表现到分割任务中。然后我们定义了一个跳跃式的架构，结合来自深、粗层的语义信息和来自浅、细层的表征信息来产生准确和精细的分割。我们的完全卷积网络成为了在PASCAL VOC最出色的分割方式（在2012年相对62.2%的平均IU提高了20%），NYUDv2，和SIFT Flow,对一个典型图像推理只需要花费不到0.2秒的时间。

## Contributions

**卷积化**

分类网络通常会在最后添加全连接层，会将原来的二维图片（矩阵）压缩成一维，从而训练出一维标量作为分类标签。而图像语义分割需要输出分割图（二维），因此将全连接层换成卷积层。

![](C:/Users/14451/Desktop/img/1.png)

**上采样**

一般的卷积神经网络使用池化层来压缩输出图片大小，而我们需要得到的是跟原图一样大小的分割图，因此，需要对最后一层进行上采样（反卷积 Deconvolution）。

caffe中，使用 **im2col**来将图片转为矩阵，使用**GEMM**来计算卷积（矩阵相乘），虽然转置卷积核卷积层一样可以训练参数，但是实际实验中并没有啥提升于是转置卷积层学习率置零了。

**跳跃结构**

直接将全卷积后的结果上采样后得到的结果很粗糙，所以，将不同池化层的结果进行上采样，结合这些结果来优化输出。pool4后面添加1x1卷积，产生附加的类别预测，将输出和在fc7经过2X上采样后的预测融合，最后用stride=16的上采样将预测变为原图像大小。把这种网结构称为FCN-16s。继续融合pool3和一个融合了 pool4和conv7的2X上采样预测，建立FCN-8s网络，融合完性能显著提升。

![](C:/Users/14451/Desktop/img/2.png)

如上图所示，对原图像进行卷积conv1、pool1后原图像缩小为1/2；之后对图像进行第二次conv2、pool2后图像缩小为1/4；接着继续对图像进行第三次卷积操作conv3、pool3缩小为原图像的1/8，此时保留pool3的featureMap；接着继续对图像进行第四次卷积操作conv4、pool4，缩小为原图像的1/16，保留pool4的featureMap；最后对图像进行第五次卷积操作conv5、pool5，缩小为原图像的1/32，然后把原来CNN操作中的全连接变成卷积操作conv6、conv7，图像的featureMap数量改变但是图像大小依然为原图的1/32,此时进行32倍的上采样可以得到原图大小,这个时候得到的结果就是叫做FCN-32s.

这个时候可以看出,FCN-32s结果明显非常平滑,不精细. 针对这个问题,作者采用了combining what and where的方法,具体来说,就是在FCN-32s的基础上进行fine tuning,把pool4层和conv7的2倍上采样结果相加之后进行一个16倍的上采样,得到的结果是FCN-16s.

之后在FCN-16s的基础上进行fine tuning,把pool3层和2倍上采样的pool4层和4倍上采样的conv7层加起来,进行一个8倍的上采样,得到的结果就是FCN-8s.

![](C:/Users/14451/Desktop/img/3.png)

## Experiment

![](C:/Users/14451/Desktop/img/4.png)

**sth good**: [FCN解析1](https://blog.csdn.net/qq_36269513/article/details/80420363)

​		  [FCN解析2](https://zhuanlan.zhihu.com/p/22976342)



# SegNet

paper:[SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561)`TPAMI 2015`

## Abstract

我们展示了一种新奇的有实践意义的深度全卷积神经网络结构，用于逐个像素的语义分割，并命名为SegNet.核心的可训练的分割引擎包含一个编码网络，和一个对应的解码网络，并跟随着一个像素级别的分类层.编码器网络的架构在拓扑上与VGG16网络中的13个卷积层相同.解码网络的角色是映射低分辨率的编码后的特征图到输入分辨率的特征图.具体地，解码器使用在相应编码器的最大池化步骤中计算的池化索引来执行非线性上采样.这消除了上采样的学习需要.上采样后的图是稀疏的，然后与可训练的滤波器卷积以产生密集的特征图.我们把我们提出的架构和广泛采用的FCN架构和众所周知的DeepLab-LargeFOV、DeconvNet架构做了比较，这种比较揭示了实现良好分割性能所涉及的内存与准确度之间的权衡。
SegNet的主要动机是场景理解应用.因此，它在设计的时候保证在预测期间，内存和计算时间上保证效率.在可训练参数的数量上和其他计算架构相比也显得更小，并且可以使用随机梯度下降进行端到端的训练.我们还在道路场景和SUN RGB-D室内场景分割任务中执行了SegNet和其他架构的受控基准测试.这些定量的评估表明，SegNet在和其他架构的比较上，提供了有竞争力的推断时间和最高效的推理内存.我们也提供了一个Caffe实现和一个web样例：<http://mi.eng.cam.ac.uk/projects/segnet/>.

## Architecture

encoder-decoder结构，encoder由VGG16的前13层卷积层组成，每个encoder层对应一个decoder层，最终解码器输出被送到多级soft max分类器，每个像素输出类概率

![](C:/Users/14451/Desktop/img/s1.png)

最大池化可以实现在输入图像上进行小的空间位移时保持平移不变性。**连续的下采样导致了在输出的特征图上，每一个像素都重叠着着大量的输入图像中的空间信息**。对于图像分类任务，多层最大池化和下采样由于平移不变性可以获得较好的鲁棒性，但导致了特征图大小和空间信息的损失。**图像分割任务中边界划分至关重要，而这么多有损边界细节的图像表示方法显然不利于分割**。因此，在进行下采样之前，**在编码器特征映射中获取和存储边界信息是十分重要的**。如果推理过程中的内存不受约束，则所有编码器特征映射(在下采样后)都可以存储。在实际应用中，情况通常不是这样，因此我们提出了一种更有效的方法来存储这些信息。它只存储最大池化索引，即存储每个池化窗口中最大特征值的位置，用于每个编码器特征映射。

![](C:/Users/14451/Desktop/img/s2.png)

对比SegNet和FCN实现Decoder的过程。SegNet保留pooling时的位置信息，upsampling时直接将数据放在原先的位置,其他位置补0，而FCN采用transposed convolutions+双线性插值，每一个像素都是运算后的结果。

**论文对解码器变种进行了详细分析和实验证明**

### BENCHMARKING

使用两个数据集来训练和测试SegNet：CamVid road scene segmentation（对自动驾驶有实际意义）和SUN RGB-D indoor scene segmentation（对AR有实际意义）

![](C:/Users/14451/Desktop/img/s3.png)

### Result

![](C:/Users/14451/Desktop/img/segnet1.gif)



sth good: [segnet解析](http://hellodfan.com/2017/11/10/%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2%E8%AE%BA%E6%96%87-SegNet/)

translate: [segnet翻译](https://blog.csdn.net/u014451076/article/details/70741629)

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

# DeepLab v1

**paper:**[DeepLabv1: Semantic image segmentation with deep convolutional nets and fully connected CRFs](<https://arxiv.org/abs/1606.00915> )`ICLR 2015`

**translate:**[deeplab v1](https://blog.csdn.net/Hanging_Gardens/article/details/78368078)

## Abstract

DCNNs近期在高级视觉任务中表现出非常好的性能，比如图像分类和目标跟踪。本文联合DCNNs和概率图模型来解决像素级分类任务,也就是语义图像分割。我们发现DCNNs最后一层的响应不能充分地用于定位精确的目标分割。这是因为存在使得DCNNs擅长高级任务的非常不变性属性。我们通过一个全连接的CRF结合DCNNs最后层的响应来克服深度网络中差的定位属性。定性的讲，我们的”DeepLab”系统能够以超过先前方法的精度来定位分割边界。定量的讲，我们的方法在PASCAL VOC2012语义图像分割任务上测试的平均IOU达到71.6%。我们展示了如何能有效地得到这些结果:注意网络的再利用；从小波社区引入”孔洞”算法，该算法在现代GPU上能以每秒8帧的速度处理神经网络响应的密集计算.

## Contributions

(1)速度: 由于“atrous”算法和感受野尺寸的减小，我们的密集DCNN以8 fps的速度运行，而完全连接的CRF的平均场推理需要0.5秒 
(2)精度: PASCAL语义分割挑战最现先进结果，超过了Mostajabi等人的方法的7.2％ 
(3)简单性: 我们的系统由两个相当完善的DCNN和CRF模块组成。

## Architecture

1.采用空洞算法的高效密集滑动窗口特征提取

由于普通下采样（最大池化）方法导致分辨率下降，局部信息丢失，想去掉池化，但是池化能使每个像素都有较大的感受野并且减少图像尺寸，因此引入空洞卷积，不进行下采样又能保证大的感受野。

![](C:/Users/14451/Desktop/img/deeplabv1-1.png)

[花样卷积示意](https://github.com/vdumoulin/conv_arithmetic)

空洞卷积 3x3 kernel dilation rate=2

![](C:/Users/14451/Desktop/img/dilation.gif)

VGG论文将7x7卷积改为3个3x3小卷积，这样做的主要目的是在保证具有相同感知野的条件下，提升了网络的深度，在一定程度上提升了神经网络的效果，同时减少了参数量。

2.**Dense CRF**:用卷积网络减少感受野并加速密集计算

- CRF在传统图像处理上主要做平滑处理。
- 但对于CNN来说，short-range CRFs可能会起到反作用，因为我们的目标是恢复局部信息，而不是进一步平滑图像。
- 引入fully connected CRF来解决这个问题，考虑全局的信息。

3.详细的边界恢复：完全连接的条件随机场

4.**多尺度预测**：将前四个最大池化层中的每一个输入图像和输出附加到一个两层MLP，特征图连接到主网络的最后一层特征图，这四个预测结果与最终模型输出拼接（concatenate）到一起，相当于多了128*5=640个channel。通过640个通道增强了馈送到分类层的聚合特征图，我们只调整新添加的权重，保留其他网络参数学习到的值。

![](C:/Users/14451/Desktop/img/deeplabv1-3.png)

5.**网络结构**

对VGG进行改进：将全连接层通过卷积层来实现；将原VGG5个最大池化中的后两个池化层stride从2变为1，相当于只进行了8倍下采样；将后两个最大池化层后的卷积层改为空洞卷积。

为了减少计算量、控制视野域，对于VGG中的第一个全连接层7x7的卷积，用3x3 或4x4代替，计算时间减少2-3倍。损失使用交叉熵之和，训练数据label对原始ground truth进行下采样8倍，预测数据label对预测结果进行双线性上采样8倍，得到预测结果。

## Experiment

![](C:/Users/14451/Desktop/img/deeplabv1-2.png)



# DeepLab V2

**paper:**[DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](<https://arxiv.org/abs/1606.00915> )`TPAMI2017`

**translate:**

## Abstract

在这项工作中，我们通过深度学习解决了图像语义分割任务，并做了三个主要贡献，实验证明了他具有实质性的实用价值。首先，我们用上采样滤波器或空洞卷积（ **atrous convolution作为密集预测任务中的强大工具**） 突出显示卷积。 Atrous卷积允许我们明确地控制在深度卷积神经网络中计算的特征响应的分辨率。它还使我们能够有效地扩大其领域 在不增加参数数量或计算量的情况下，过滤器的视图可以包含更多的上下文。第二，我们提出了一个**空洞空间金字塔池（ASPP）以多尺度的信息得到更强健的分割结果 **。 SPP并行的采用多个采样率的空洞卷积层来探测，从而以多尺度捕获对象以及图像上下文。第三，我们通过结合DCNN和概率图形的方法来改进对象边界的定位。 DCNN中通常的最大池和下采样的组合实现了平移不变性，但却对定位精度有一定的影响。我们通过将最终DCNN层的响应与完全连接的CRF相结合来克服这个问题，定性和定量显示都提高了定位性能。我们的建议 “DeepLab”系统在PASCAL VOC-2012语义图像分割任务中设置了新的最新技术水平，达到了79.7％的mIOU
测试集，并将结果推进到其他三个数据集：PASCAL-Context，PASCAL-Person-Part和Cityscapes。

## Contributions

将DCNN应用在语义分割上，主要有三个问题：降低特征分辨率、多个尺度上存在对象、由于DCNN的内在不变性 定位精度变低。第一个问题因为DCNN连续的最大池化和下采样组合引起的空间分辨率下降，为了解决这个问题，deeplabv2在最后几个最大池化层中除去下采样，使用空洞卷积以分高的采样密度计算特征映射。第二个问题因为在多尺度上存在物体，解决办法是将一张图缩放不同版本，汇总特征或最终预测map得到结果，但这个增加了计算特征响应，需要大量存储空间，受到空间金字塔池化（SPP）启发，提出了一个类似结构，在给定的输入上以不同采样率的空洞卷积并行采样，相当于以多个比例捕捉图像的上下文，称为空洞空间金字塔池化（ASPP）模块。第三个问题涉及到对象分类要求空间变换不变性，这影响了DCNN的空间定位精度，解决办法是在计算最终分类结果时使用跳跃层，将前面的特征融合到一起。

![](C:/Users/14451/Desktop/img/v2-1.png)

- 输入经过改进的DCNN(带空洞卷积和ASPP模块)得到粗略预测结果，即`Aeroplane Coarse Score map`
- 通过双线性插值扩大到原本大小，即`Bi-linear Interpolation`
- 再通过全连接的CRF细化预测结果，得到最终输出`Final Output`

DeepLabv2的主要优点在于：

速度： DCNN在现代GPU上以8FPS运行，全连接的CRF在CPU上需要0.5s
准确性：在PASCAL VOC2012,PASCAL-Context, PASCALPerson-Part,Cityscapes都获得的优异的结果
简单性：系统是由两个非常成熟的模块级联而成，DCNNs和CRFs

本文DeepLabv2是在DeepLabv1的基础上做了改进，基础层由VGG16换成了更先进的ResNet，添加了多尺度和ASPP模块技术得到了更好的分割结果。

![](C:/Users/14451/Desktop/img/v2-2.png)



### Experiment

![](C:/Users/14451/Desktop/img/v2-3.png)



![](C:/Users/14451/Desktop/img/v2-4.png)



sth good: [deeplabv2](https://blog.csdn.net/u011974639/article/details/79138653)

# DeepLab v3

**paper:** [DeepLabv3:Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)  `ECCV2018`

## Abstract

空间金字塔池模块或编码 - 解码器结构用于深度神经网络中解决语义分割任务。前一种网络能够通过利用多个速率和多个有效视场的过滤器或池化操作探测输入特征来编码多尺度上下文信息，而后一种网络可以通过逐渐恢复空间信息来捕获更清晰的对象边界。在这项工作中，我们建议结合两种方法的优点。具体来说，我们提出的模型DeepLabv3 +通过添加一个简单而有效的解码器模块来扩展DeepLabv3，以优化分割结果，尤其是沿着对象边界。我们进一步探索Xception模型并将深度可分离卷积应用于Atrous Spatial Pyramid
Pooling和解码器模块，从而产生更快更强的编码器-解码器网络。我们证明了所提出的模型在PASCAL VOC 2012语义图像分割数据集上的有效性，并且在没有任何后处理的情况下在测试集上实现了89％的性能。我们的论文附有Tensorflow中提出的模型的公开参考实现。

## Contributions

•我们提出了一种新颖的编码器-解码器结构，它采用DeepLabv3作为功能强大的编码器模块和简单而有效的解码器模块。

•在我们提出的编码器 - 解码器结构中，可以通过atrous卷积任意控制提取的编码器特征的分辨率，以折中精度和运行时间，这对于现有的编码器解码器模型是不可能的。

•我们将Xception模型用于分割任务，并将深度可分离卷积应用于ASPP模块和解码器模块，从而产生更快更强的编码器-解码器网络。

•我们提出的模型在PASCAL VOC 2012数据集上获得了新的最新性能。我们还提供设计选择和模型变体的详细分析。

•我们公开提供基于Tensorflow的提议模型实现。

