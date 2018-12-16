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
