# Paper:Beyond Short Snippets: Deep Networks for Video Classification,arXiv:1503.08909v2 [cs.CV] 13 Apr 2015

## Abstract

Convolutional neural networks (CNNs) have been extensively applied for image recognition problems giving stateof-the-art results on recognition, detection, segmentation
and retrieval. In this work we propose and evaluate several
deep neural network architectures to combine image information across a video over longer time periods than previously attempted. We propose two methods capable of handling full length videos. The first method explores various
convolutional temporal feature pooling architectures, examining the various design choices which need to be made
when adapting a CNN for this task. The second proposed
method explicitly models the video as an ordered sequence
of frames. For this purpose we employ a recurrent neural
network that uses Long Short-Term Memory (LSTM) cells
which are connected to the output of the underlying CNN.
Our best networks exhibit significant performance improvements over previously published results on the Sports 1 million dataset (73.1% vs. 60.9%) and the UCF-101 datasets
with (88.6% vs. 88.0%) and without additional optical flow
information (82.6% vs. 73.0%).

### Contribution
![image](image/$O$)D@K(LH)M~`X((K8$Q7D.png)


CNN在静态图像的处理中大显神通，但是在处理视频信息时，却是力有不逮。这主要是有两个原因。\
1.CNN处理静态图像的时候丢失了许多视频帧之间的暂态信息。\
2.处理一个短视频也需要许多计算资源。\
基于此，本论文将CNN与LSTM结合在一起对视频数据进行特征提取，单帧的图像信息通过CNN获取特征，然后将CNN的输出按时间顺序通过LSTM，
最终将视频数据在空间和时间维度上进行了特征表达。在这个过程中引入了光流这一个概念，为以后视频处理提供了范式。

# Paper：
