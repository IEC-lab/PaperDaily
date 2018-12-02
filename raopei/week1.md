# paper_reading

## paper:ImageNet Classification with Deep Convolutional Networks ,NIPS2012

### Abstract

We trained a large, deep convolutional neural network to classify the 1.2 million
high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different
classes. On the test data, we achieved top-1 and top-5 error rates of 37.5%
and 17.0% which is considerably better than the previous state-of-the-art. The
neural network, which has 60 million parameters and 650,000 neurons, consists
of five convolutional layers, some of which are followed by max-pooling layers,
and three fully-connected layers with a final 1000-way softmax. To make training
faster, we used non-saturating neurons and a very efficient GPU implementation
of the convolution operation. To reduce overfitting in the fully-connected
layers we employed a recently-developed regularization method called “dropout”
that proved to be very effective. We also entered a variant of this model in the
ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%,
compared to 26.2% achieved by the second-best entry.

### Contribution

可算是深度学习的起源,极大地推动对神经网络的研究，在论文中，作者自我总结了一下几点：
1.使用ReLU作为激活函数。
2.利用GPU并行计算。
3.提出了LRN层。
4.在CNN中使用重叠的最大池化


## paper：Very Deep Convolutional Networks For Large-Scale Image Recognition，ICLR 2015

### Abstract

In this work we investigate the effect of the convolutional network depth on its
accuracy in the large-scale image recognition setting. Our main contribution is
a thorough evaluation of networks of increasing depth using an architecture with
very small (3×3) convolution filters, which shows that a significant improvement
on the prior-art configurations can be achieved by pushing the depth to 16–19
weight layers. These findings were the basis of our ImageNet Challenge 2014
submission, where our team secured the first and the second places in the localisation
and classification tracks respectively. We also show that our representations
generalise well to other datasets, where they achieve state-of-the-art results. We
have made our two best-performing ConvNet models publicly available to facilitate
further research on the use of deep visual representations in computer vision.

### Contribution

强调卷积网络的深度，CNN必须保证拥有一个足够深的网络结构来体现其处理视觉信息的层次性。

### 评价

这两篇论文皆属于某些方面的开山之作，文章写得很简练，但里面的道理与妙处我尚不能很好地理解与体会，只能掌握比较直观的模型，知道这两个
网络怎么实现。怎么说，有点像黑风双煞看九阴真经，经文中奥妙之处看不出来，就学得会一些下流拳脚功夫，还需磨练。
