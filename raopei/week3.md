# Paper:Generative Adversarial Nets,arXiv:1406.2661v1

## Abstract

We propose a new framework for estimating generative models via an adversarial
process, in which we simultaneously train two models: a generative model G
that captures the data distribution, and a discriminative model D that estimates
the probability that a sample came from the training data rather than G. The training
procedure for G is to maximize the probability of D making a mistake. This
framework corresponds to a minimax two-player game. In the space of arbitrary
functions G and D, a unique solution exists, with G recovering the training data
distribution and D equal to 1
2
everywhere. In the case where G and D are defined
by multilayer perceptrons, the entire system can be trained with backpropagation.
There is no need for any Markov chains or unrolled approximate inference networks
during either training or generation of samples. Experiments demonstrate
the potential of the framework through qualitative and quantitative evaluation of
the generated samples.

### Contribution

利用一个生成器G和一个鉴别器D，G输入一个噪声样本，然后把它包装成一个逼真的样本，也就是输出，
D来判断输入的样本是真是假。单独交替迭代训练，使两者都得到训练，G希望以假乱真，使D无法分辨
输出是否为真，D则希望去伪存真，可以准确分辨真样本集和G生成的样本集，这就是对抗训练的思想。


# Paper:Spatial Transformer Networks,arXiv:1506.02025v3

## Abstract

Convolutional Neural Networks define an exceptionally powerful class of models,
but are still limited by the lack of ability to be spatially invariant to the input data
in a computationally and parameter efficient manner. In this work we introduce a
new learnable module, the Spatial Transformer, which explicitly allows the spatial
manipulation of data within the network. This differentiable module can be
inserted into existing convolutional architectures, giving neural networks the ability
to actively spatially transform feature maps, conditional on the feature map
itself, without any extra training supervision or modification to the optimisation
process. We show that the use of spatial transformers results in models which
learn invariance to translation, scale, rotation and more generic warping, resulting
in state-of-the-art performance on several benchmarks, and for a number of
classes of transformations.

### Contribution

主要贡献是引入了一个空间变换模块，以一种方式转换输入图像，使得随后的图层更容易进行分类。
该模块包括：\
1.一个本地化网络，应用参数有输入量和空间变换输出。 对于仿射变换（affine transformation），参数 θ可以是6维的。\
2.创建一个采样网格。这是通过在本地化网络中用创建的仿射变换theta来扭曲常规网格的结果。\
3.一个采样器，其目的是对输入特征图做扭曲变换。\
本篇文章展示了CNN网络的一种改进方式：不一定需要对架构进行改变，而可以通过对输入图像进行仿射变换，使得模型对转换、
缩放、旋转操作变得更加稳定。
