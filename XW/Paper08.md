# **Detection**

## 1.Towards Adversarially Robust Object Detection 

https://arxiv.org/pdf/1907.10786.pdf

将detection 看作多任务学习,分析了各类对抗攻击方式对task losses的影响,提出了面向对抗攻击的鲁棒detection.

## 2.A Survey of Deep Learning-based Object Detection

https://arxiv.org/pdf/1907.09408.pdf

对深度学习做Object detection的综述,涵盖多种主流架构及不同configuration的在COCO 2017上的比较


## 3.DR Loss: Improving Object Detection by Distributional Ranking

https://arxiv.org/abs/1907.10156

修改loss函数处理样本不平衡问题,通过只修改loss在COCO数据集上把mAP从39.1%提高到41.1%

## 4.BASNet: Boundary-Aware Salient Object Detection

http://openaccess.thecvf.com/content_CVPR_2019/papers/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.pdf

https://github.com/NathanUA/BASNet

关注边界的显著性检测,Loss设计上,使用了交叉熵,结构相似性损失,IoU 损失这三种的混合损失,使网络更关注于边界质量,而不是像以前那样只关注区域精度

## 5. It’s All About The Scale - Efficient Text Detection Using Adaptive Scaling

https://arxiv.org/pdf/1907.12122.pdf

针对text detection任务中,text在图片里面scale不同的问题,以往的工作都是用放大的图片当input/多张不同尺度缩放的图片当input,造成计算量大,这篇工作先在缩小的照片预测出text segmentation和scale,再将不同scale的text region缩放到适合尺寸送到Text extractor中,实验表明此方法效果提升显著（特别是在小图input上）


# **Segmentation**

## 1.ET-Net: A Generic Edge-aTtention Guidance Network for Medical Image Segmentation

https://arxiv.org/pdf/1907.10936.pdf

MICCAI 2019,天津大学的文章,值得参考的地方是用了edge-attention提升segmentation性能,通用在各个医疗图片分割任务上,作者申称会公布代码放在 https://github.com/ZzzJzzZ/ETNet

## 2.Interpretability Beyond Classification Output: Semantic Bottleneck Networks

https://arxiv.org/pdf/1907.10882.pdf

旨在为deep segmentation 架构提供可解释性,文章在传统segmentation network中加入Semantic Bottleneck(SB),,输出低维的semantic concept. 比较亮点的是加入SB结构后仍然能达到SOTA performance, 同时SB结构使得error case study变得容易,利用SB结构输出,也可以对segmentation结果区间进行置信度打分

# **Convolution**

## 1.MixNet: Mixed Depthwise Convolutional Kernels

https://arxiv.org/abs/1907.09595v1

https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet

提出新的convolution机制MDConv, 可与普通cnn替换,用于多种任务,使用autoML和MDConv, 提出新模型MixNets,MixNets 在COCO object detection超过基线模型2%, ImageNet under mobile settings做到SOTA

# **Optimizer**

## 1.Lookahead Optimizer: k steps forward, 1 step back

https://arxiv.org/abs/1907.08610v1

https://github.com/alphadl/lookahead.pytorch

多伦多大学向量实验室论文，adam原作和hinton为2,3作以前的optimizer论文着重于利historical gradients自适应学习率 如Adam/AdamGradaccelerated scheme 如 Nesterov momentum/ Polyak heavy-ball作者提出方法与上述方法正交维护两套参数.一次更新:快参数<--慢参数,快参数多次更新后,慢参数朝快参数方向更新一次,作者称此方法超参数少,训练robust. 在多数据集上接近或是超过Adam/SGD+momentum