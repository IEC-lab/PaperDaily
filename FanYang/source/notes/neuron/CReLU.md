# CReLU

## 论文

"Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units", ICML 2016, [paper](https://arxiv.org/pdf/1603.05201)

## 摘要

最近，卷积神经网路 (CNN) 已被用作解决机器学习和计算机视觉等许多问题的强大工具。本文旨在提供对卷积神经网络特性的深入理解，以及提升许多 CNN 架构性能的通用方法。具体而言，我们首先考察了现有的 CNN 模型，并观察到一个有趣的特性：较低层的卷积滤波器成对出现（即具有相反的滤波器）。收到这种观察的启发，我们提出了一种新颖、简单而有效的激活方案，称为级联 ReLU (CReLU)，并在理论上分析器在 CNN 中的重构特性。我们将 CReLU 集成到几个最先进的 CNN 架构中，减少了可训练参数，并在 CIFAR-10/100 和 ImageNet 数据集上得到了识别性能提升。我们的结果表明，在更好地理解 CNN 的情况下，通过简单的改进，可以获得显著的性能提升。

## 观察与分析

<center>![](/_static/img/Neuron_Act_CReLU_0.png)<br/>图 1 AlexNet 第一层卷积滤波器可视化 </center>

通过对 AlexNet 第一层归一化的滤波器进行可视化，发现低层滤波器呈现“成对”的有趣现象，即对任意滤波器，总存在一个几乎和它相反的滤波器。更确切地说，加入某个滤波器的单位长度向量为 `$\phi_i$`，定义它的成对（相反）滤波器为 `$\overline{\phi_i}=\arg\min_{\phi_j} \left \langle \phi_i, \phi_j \right \rangle$`，同时定义其余弦相似度 `$\mu_i^\phi=\left \langle \phi_i, \overline{\phi_i} \right \rangle$`。

<center>![](/_static/img/Neuron_Act_CReLU_1.png)<br/>图 2 AlexNet 训练的的和随机的卷积滤波器的余弦直方图分布</center>

为了系统地研究更高层中的成对现象，对 conv1-conv5 绘制了如图 2 余弦直方图。其中蓝色的为训练后的卷积滤波器的余弦直方图，红色的是随机高斯初始化的余弦直方图。可以看出，随机初始化呈现出尖锐的以 0 为中心的分布，而经过训练的卷积滤波器呈现以负数为中心，且大幅度偏向于负数区域的现象，说明“成对”现象在较低层卷积滤波器中普遍存在。随着卷积层数的升高，这种分布偏向逐渐减弱，直到在 conv5 中几乎消失，说明“成对”现象在较高层卷积滤波器中逐渐减少。

从这种观察中，我们得到一种假设：<font color=#FF3E96>尽管 ReLU 消除了线性响应的负数部分，但是 CNN 的前几层试图通过学习成对（负相关）的滤波器来同时捕获正相和负相的信息。</font>这意味着我们可以利用这种成对先验，设计一种同时允许正负激活的方法，从而减轻由 ReLU 非线性引起的在卷积滤波器中的冗余问题，更有效地利用可训练参数。为此，我们提出了一种新的激活方案，CReLU (Concatenated ReLU)。

Caffe prototxt for CReLU and Keras AntiRectifier<sup>[1]</sup> [模块可视化](https://ethereon.github.io/netscope/#/gist/155019b053b3b93fe1822b7fd84935c0)
<script src="https://gist.github.com/yangfly/155019b053b3b93fe1822b7fd84935c0.js"></script>

## 实验

对比一半通道的 CReLU 和普通 ReLU 及 AVR (绝对值 ReLU)

<center>表 1 CIFAR-10/100 对比实验<br/>![](/_static/img/Neuron_Act_CReLU_2.png)</center>

<center>表 2 ImageNet 对比实验<br/>![](/_static/img/Neuron_Act_CReLU_3.png)</center>

## 引用

[1]: [Keras AntiRectifier](https://github.com/keras-team/keras/blob/master/examples/antirectifier.py)<br/>