# Dataset

- **FG-Net**:1002 images
- **MORPH1**:1690 images

- **MORPH2**:55608 images、unbalanced ethnic(96% African and European ,less than 1% Asian)
- **AFAD**:160K Asian facial images

# Ordinal Regression

**paper：**[Ordinal Regression With Multiple Output CNN for Age Estimation](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Niu_Ordinal_Regression_With_CVPR_2016_paper.pdf)`CVPR2016`

**code:**[Ordinal Regression](https://github.com/luoyetx/OrdinalRegression)

## Abstract



## Contributions

- 利用端到端的深度学习方法解决序数回归问题
- released a dataset named AFAD , the largest public dataset to date

## Results

# RankingCNN

**paper**：[Using Ranking-CNN for Age Estimation](http://openaccess.thecvf.com/content_cvpr_2017/poster/2148_POSTER.pdf)`CVPR2017`

**code:** [Using-Ranking-CNN-for-Age-Estimation](https://github.com/RankingCNN/Using-Ranking-CNN-for-Age-Estimation)

## Abstract

## Contributions

## Results

# SSR-Net

**paper**：[SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation](https://www.csie.ntu.edu.tw/~cyy/publications/papers/Yang2018SSR.pdf)`IJCAI2018`

**code:** [SSRNet](https://github.com/shamangary/SSR-Net)

## Abstract

## Contributions

新的人臉影像年紀估測模型，誤差 3.16 歲，與最好的模型差距 0.5 歲，但參數不到其1/1000，整個模型參數僅0.3MB，非常適合用於嵌入式系統

## Results

# DeepRegressionForest

**paper:**[Deep Regression Forests for Age Estimation](https://arxiv.org/abs/1712.07195)`CVPR2018`

**code**:[caffe-DeepRegressionForests](https://github.com/shenwei1231/caffe-DeepRegressionForests)

## Abstract

不同的人在相同的年纪样貌有很大差异，所以年龄是异构数据。提出`Deep Regression Forests(DRFs)`算法，可以端到端的学习，预测年龄。DRFs将分裂节点连接在CNN的全连接层后面，通过联合学习来处理异构数据，在分裂节点进行输入相关的data partitions，在叶节点上进行data abstractions。这种联合学习遵循一种`交替策略`：首先固定叶节点，分裂节点和CNN的参数通过back-propagation 优化，然后，固定分裂节点，叶节点通过从变分边界导出的step-size free和快速收敛的更新规则进行优化。

年龄预测主要分为两类任务：1. 真实年龄预测 2. 年龄区间预测。 面临的主要困难是年龄是异质的：1. 同一个年纪的不同人外表呈现差异很大 2.同一个人在不同的年龄面部也会发生很大变化。对异质数据的建模，现在的年龄估计方法要么是找到一个基于内核的全局非线性映射，或者利用分而治之的策略来划分数据空间并学习多个局部回归器。但是它们都各自存在缺点：学习不稳定内核不可避免地收到异质数据分布偏差地影响，因此容易导致过拟合；分而治之是一种学习不稳定年龄变化的很好的策略，但是现有的方法是make hard partitions ，所以，可能找不到同类子集来学习局部回归器。

为了解决上述问题，提出不同的`回归森林`用于年龄估计。随机森林或者随机决策树，是一种常用的ensemble predictive model，每棵树在分裂节点做data partition ,在叶节点做data abstraction。传统的回归森林根据启发式方法做hard data partitions，在每个分裂节点上进行局部最优的hard decisions。和传统的方法不同的是，本文提出一个`可微分回归森林`方法，做soft data partitions，这样一个依赖输入的partition函数就可以学会处理异质数据。另外，输入特征空间和叶节点的data abstractions（局部回归器）可以联合学习，保证了局部的输入-输出的相关性在叶节点是同质的。我们的回归森林可以

## Contributions

## Results

#  Mean-Variance Loss

**paper**:[Mean-Variance Loss for Deep Age Estimation From a Face](http://openaccess.thecvf.com/content_cvpr_2018/papers/Pan_Mean-Variance_Loss_for_CVPR_2018_paper.pdf)`CVPR2018`

**code:**-

## Abstract

## Contributions

## Results

