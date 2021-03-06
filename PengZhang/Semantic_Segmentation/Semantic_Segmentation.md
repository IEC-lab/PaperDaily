
## <center> Light-Weight RefineNet for Real-Time Semantic Segmentation [PDF](https://arxiv.org/abs/1810.03272)</center>

### <center> Abstract </center>

   We consider an important task of effective and efficient semantic image segmentation. In particular, we adapt a powerful semantic segmentation architecture, called RefineNet, into the more compact one, suitable even for tasks requiring real-time performance on high-resolution inputs. To this end, we identify computationally expensive blocks in the original setup, and propose two modifications aimed to decrease the number of parameters and floating point operations. By doing that, we achieve more than twofold model reduction, while keeping the performance levels almost intact. Our fastest model undergoes a significant speed-up boost from 20 FPS to 55 FPS on a generic GPU card on 512x512 inputs with solid 81.1% mean iou performance on the test set of PASCAL VOC, while our slowest model with 32 FPS (from original 17 FPS) shows 82.7% mean iou on the same dataset. Alternatively, we showcase that our approach is easily mixable with light-weight classification networks: we attain 79.2% mean iou on PASCAL VOC using a model that contains only 3.3M parameters and performs only 9.3B floating point operations. 

![pic](http://5b0988e595225.cdn.sohucs.com/images/20181129/2eafaf2db46b47aeae03a9635dce4fd5.jpeg)

本文是阿德莱德大学发表于 BMVC 2018 的工作，论文关注的问题为实时语义分割。作者将源自 CVPR 2017 的 RefineNet 作为基础网络，结合两种残差模块 RCU（残差卷积单元）和 CRP（链式残差池化）有效减少模型参数和计算量，在 512 x 512 大小的输入图像上将分割速度从 20 FPS 显著提升至 55 FPS。<br />
该文的目的很简单，在CVPR2017的RefineNet语义分割算法基础上减少模型参数和计算量。
 <br />RefineNet使用经典的编码器-解码器架构，CLF为3*3卷积，卷积核个数为语义类的个数，编码器的骨干网可以是任意图像分类特征提取网络，重点是解码器部分含有RCU、CRP、FUSION三种重要结构。

RCU即residual convolutional unit（残差卷积单元），为经典残差网络ResNet中的residual block去掉batch normalisation部分，由ReLU和卷积层构成。

CRP为链式残差池化（chained residual pooling），由一系列的池化层与卷积层构成，以残差的形式排列。

RCU与CRP中使用3*3卷积和5*5池化。

FUSION部分则是对两路数据分别执行3*3卷积并上采样后求和SUM。

1）替换3*3卷积为1*1卷积

虽然理论3*3卷积理论上有更大的感受野有利于语义分割任务，但实际实验证明，对于RefineNet架构的网络其并不是必要的。

2）省略RCU模块

作者尝试去除RefineNet网络中部分及至所有RCU模块，发现并没有任何的精度下降，并进一步发现原来RCU blocks已经完全饱和。
