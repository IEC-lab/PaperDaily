# 人脸：DSFD 

## 论文

"DSFD: Dual Shot Face Detector", Arxiv 2018, [paper](https://arxiv.org/pdf/1810.10220.pdf)

## 摘要

卷积神经网络（CNN）最近在人脸检测方面取得了巨大成功。然而，由于尺度，姿势，遮挡，表情，外观和照明的高度可变性，人脸检测对当前的检测方法而言依然是非常具有挑战性的问题。在本文中，我们提出了一种新的人脸检测网络：DFSD，在 SSD 的基础上引入特征增强模块（FEM），用于特征传递以将单筒检测器扩展到双筒检测器。特别是采用两组锚框计算的锚框提升损失（Progressive Anchor Loss，PAL）来有效促进该特征。另外，我们通过整合新颖的数据增强技术和锚框设计策略，提出了一种改进的锚点匹配方法（IAM），为回归问题提供更好的初始化。在主流基准测试集上进行广泛实验：WIDER FACE（Easy：0.966，Medium：0.957，Hard：0.904）和 FDDB（discontinuous：0.991，continuous：0.862）证明了 DSFD 优于最先进的人脸检测器。

## 回顾

人脸检测器依旧存在的问题：
1. 特征学习：特征提取部分对于人脸检测器至关重要。但广泛使用的 FPN 只是聚合了上下层的层次特征图以获得丰富的特征，却没有考虑当前层信息，且忽略了锚框间的上下文关系。
2. 损失设计：常规损失 Softmax Loss 和 Smooth-L1 loss，以及为解决类别不均衡问题的 Focal Loss 和适用于特征金字塔的 Hierarchical loss<sup>[1]</sup>，都未考虑提升不同特征层的学习能力。
3. 锚框匹配：检测器在不同层特征上定义了不同尺度和长宽比的预设锚框，而且采用锚框补偿的增加小脸正锚框数。但连续的人脸尺度和离散的锚框尺度导致正负样本比例的巨大差异。

论文主要贡献：
- FEM：一种新颖的特征增强模块，能够利用不同层次的信息，从而获得更多的可辨性和鲁棒性特征。
- PAL：通过一系列较小的锚框将辅助监督进入到前面的层，从而有效促进低层特征。
- 锚框匹配：改进的锚框匹配策略，尽可能低匹配锚框和标注脸，为回归器提供更好的初始化。
- 最佳性能：在流行的主流基准测试集 FDDB 和 WIDER FACE 上均取得了最好的成绩。

## 网络框架

<center>![](/_static/img/Detect_Face_DSFD_0.png)图 1 DSFD 框架在 VGG16 架构上使用特征增强模块 (b)，从原始特征 (a) 生成增强特征 (c)，并连接不同的损失。</center>

如图 1 所示，DSFD 使用和 PyramidBox<sup>[2]</sup> 和 S3FD<sup>[3]</sup> 一样的 VGG16 主干，延伸出原始的六个特征检测层，并通过特征增强模块转化出六个相同尺寸的增强特征检测层。不同的是，使用 FEM 中的感受野增强和新的锚框设计策略后，步长、锚框和感受野不需要满足等比原则。

1. 特征增强模块：Feature Enhance Module，上层特征放大后与下层特征融合（按位乘？），然后接感受野增强：三路空洞卷积。
  <center>![](/_static/img/Detect_Face_DSFD_1.png)<br/>图 2 特征增强模块</center>

2. 锚框提升损失：Progressive Anchor Loss，双筒均使用多目标损失 ( Softmax 和 Smooth-L1)， 区别是第一筒相对于第二筒使用较小的锚框。
  <br/>第一筒损失： ![](/_static/img/Detect_Face_DSFD_2.png)
  <br/>双筒损失： ![](/_static/img/Detect_Face_DSFD_3.png)
  <center>表 1 双筒不同的锚框设置<br/>![](/_static/img/Detect_Face_DSFD_4.png)</center>

3. 改进锚框匹配：
  锚框设计：长宽比 1.5:1 (基于统计)，第一筒同级锚框尺寸是第二筒的一半。
  数据采样：2/5 采用 PyramidBox-style：data-anchor-sampling，3/5 采用 SSD-style。
  匹配阈值：IoU=0.4
  预测配置：检测输出 top-5000，经过 NMS-0.3 输出最终 top-750。
  <center>![](/_static/img/Detect_Face_DSFD_5.png)<br/>图 3 采样人脸的尺度分布，左原始 SSD-style，右综合版，可见增加了锚框尺寸附近的样本</center>

## 实验

总体上，虽然在 Wider Face 上获得了最好的结果，但是提出的创新过于粗糙，不足以支撑论文的论点，性能提升的主要贡献在于使用了更加复杂的主网络 Res152。可能最大的贡献在于，使我们更加坚定了 **大力出奇迹**。

<center>表 2 特征增强模块 FEM 的提升<br/>![](/_static/img/Detect_Face_DSFD_6.png)</center>
> 感觉没有惊喜也不值得。

<center>表 3 双筒 PAL 的提升 <br/>![](/_static/img/Detect_Face_DSFD_7.png)</center>
> 感觉没有惊喜，相对比 SRN。

<center>表 4 改进锚框匹配 IAM 的提升 <br/>![](/_static/img/Detect_Face_DSFD_8.png)</center>
> 感觉没有惊喜，不算创新。

<center>表 5 网络框架的提升 <br/>![](/_static/img/Detect_Face_DSFD_9.png)</center>
> 在 ImageNet 上分类越高的，不一定在 Wider Face 上越好。


