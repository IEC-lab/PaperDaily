# 人脸：SRN

## 论文

"Selective Refinement Network for High Performance Face Detection", AAAI 2019, [paper](https://arxiv.org/pdf/1809.02693)

## 摘要

高性能人脸检测仍然是一个非常具有挑战性的问题，尤其是当存在许多小脸时。本文提出了一种新颖的单段人脸检测器：选择性精细网络（Selective Refinement Network, SRN），它将新型的两段分类和回归操作选择性地引入基于锚框的人脸检测器中，以减少误报并同时提高定位精度。特别是，SRN 由两个模块组成：选择性两步分类（STC）模块和选择性两步回归（STR）模块。 STC 旨在从低检测层过滤掉大多数简单的负锚，以减少后续分类器的空间，而 STR 设计用于从高检测层粗略调整锚框的位置和大小，以提供更好的初始化随后的回归量。此外，我们设计了一个感受野增强（RFE）块，以提供更多样化的感受野，这有助于更好地捕捉一些极端姿势的人脸。因此，所提出的SRN检测器在所有广泛使用的面部检测基准上实现了最先进的性能，包括 AFW，PASCAL 人脸，FDDB 和 WIDER FACE 数据集。我们将开放源码以促进对面部检测问题的进一步研究。
  
## 回顾

人脸检测器改进空间：
1. 召回效率：高召回率下减少误报数量，分类要准，减少 False Positive。
2. 定位精度：因 Wider Face 采用 MSCOCO 的评估标准，要提高边框位置的准确性。

<center>![](/_static/img/Detect_Face_SRN_0.png)图 1 (a) STC 和 STR 分别增加正/负样本比例 38 和 3 倍；(b) SRN 增加大概 20% 的召回率；<br/>(c) STR 提供更好的回归初始框；(d) SRN 相对于 RetinaNet 定位更准 </center>

## 网络框架

SRN 整体上是基于 RefineDet<sup>[1]</sup> 和 RetinaNet<sup>[2]</sup> 结合的改进。

<center>![](/_static/img/Detect_Face_SRN_1.png)图 2 SRN 的网络结构，包括 STC，STR，RFM </center>

1. STC：Selective Two-Step Classification，由 RefineDet<sup>[1]</sup> 引入，通过级联的两步分类机制，在第一段采用预设阈值 `$\theta=0.99$` 提前清楚简单背景锚框。与 RefineDet 不同的是，SRN 只在低三段使用了两步分类，因为低三层产生了绝大多数（88.9%）的锚框和更严重的类别不均衡，同时面临着特征不充分的问题，需要使用两步分类来减少搜索空间、缓解类别不均衡、二段精细分类。损失函数：
  <center>![](/_static/img/Detect_Face_SRN_2.png)</center>
  <center>表 1 两步分类作用于不同金字塔层次上的效果<br/>![](/_static/img/Detect_Face_SRN_3.png)</center>

2. STR：Selective Two-Step Regression，单步回归在 MSCOCO 类型的评估标准上是不够准的，最新研究 RefineDet<sup>[1]</sup> 和 Cascade R-CNN<sup>[3]</sup> 均采用级联结构来提升定位精度。但是盲目地向特定任务（即人脸检测）添加多步回归通常会适得其反。实验结果（表 2）表明 STR 只对高三层有积极意义。这背后的原因来自于两方面：1）低层锚框特征粗糙难以进行两步回归；2）在训练阶段，如果过渡关注低层的困难的回归任务，导致更大的回归损失阻碍了更重要的分类任务。损失函数：
  <center>![](/_static/img/Detect_Face_SRN_4.png)</center>
  <center>表 2 两步回归作用于不同金字塔层次上的效果<br/>![](/_static/img/Detect_Face_SRN_5.png)</center>

3. 感受野增强模块：Receptive Field Enhancement，检测器特征提取模块通常都是方形感受野，感受野的单一性影响对具有不同长宽比的物体的检测，例如 Wider Face 训练集中有相当一部分脸部长宽比大于 2 或小于 0.5。为了解决网络感受野与面部长宽比之间的不匹配问题，使用感受野增强模块（RFE）替代 RetinaNet<sup>[2]</sup> 中的 class subnet 和 box subnet，以增加感受野的多样化。
  <center>![](/_static/img/Detect_Face_SRN_6.png)<br/>图 2 RFE 模块结构</center>

## 训练和推理

- 数据扩充：类似于 RefineDet<sup>[1]</sup>，随机光学失真，随机扩展、切割和镜像。
- 主干网络：ResNet-50<sup>[4]</sup> 加上 6 个特征金字塔结构作为主干，旁路的自顶向下类似于 FPN<sup>[5]</sup>。
- 专用模块：STC 应用于低三层，STR 应用于高三层，RFE 代替 class/box subnet。
- 锚框设置：尺度=步长`$\times(2,2\sqrt{2})$`， 高宽比=1.25， 覆盖尺度范围 8-362 像素。
- 锚框匹配：`$\geq \theta_p$` 为正，`$\lt \theta_n$` 为负，中间忽略；第一段 `$\{0.3,0.7\}$`，第二段 `$\{0.4,0.5\}$`。
- 难负例挖掘：使用 Focal Loss<sup>[2]</sup>，因此省略。
- 损失函数：Focal Loss<sup>[2]</sup> + Smooth-L1 Loss
- 优化：初始化 “xavier”，batch-size=32，初始学习率 0.001
- 推理：STC 滤掉简单背景锚框 `$\theta=0.99$`，STR 输出 top-2000，经过 NMS-0.5 输出最终 top-750。

## 实验

<center>表 3 消融实验，三种专用模块的作用<br/>![](/_static/img/Detect_Face_SRN_7.png)</center>

<center>表 4 STC：不同召回率下的 False Positive 个数<br/>![](/_static/img/Detect_Face_SRN_8.png)</center>

<center>表 5 STR：在 Wider Face Hard set 上不同 IoU 阈值下的 AP<br/>![](/_static/img/Detect_Face_SRN_9.png)</center>


## 引用

[1]: [CVPR 2018: Single-Shot Refinement Neural Network for Object Detection](../../RefineDet.html)<br/>
[2]: [ICCV 2017: Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)<br/>
[3]: [CVPR 2018: Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/pdf/1712.00726.pdf)<br/>
[4]: [CVPR 2016: Deep residual learning for image recognition](https://arxiv.org/pdf/1512.03385.pdf)<br/>
[5]: [CVPR 2017: Feature pyramid networks for object detection](https://arxiv.org/pdf/1612.03144.pdf)<br/>