检测技术
========

**通用目标检测**

- RefineDet
   "Single-Shot Refinement Neural Network for Object Detection"  
    推荐，一个二次回归的单段检测器，相对于 SSD，速度下降不大，但准确率显著提升。
  
- Detect at 200 FPS
   "Object detection at 200 Frames Per Second"
    推荐，一个二次回归的单段检测器，相对于
    

**人脸检测**

- SRN: Seletive Refinement Network
   "Selective Refinement Network for High Performance Face Detection"
    推荐，根据 RefineDet 思路发展而来的人脸检测器，效果很好。

- DSFD: Dual Shot Face Detector
   "DSFD: Dual Shot Face Detector"
    不推荐，虽然目前性能最好，但提升主要来自于使用复杂网络。

- FaceBoxes: SSD-style 的快速检测器 
   "FaceBoxes: A CPU Real-time Face Detector with High Accuracy"
    最小脸 20px，20FPS@CPU，125FPS@GPU。

- DCFPN：Single Head 快速检测器
   "Detecting Face with Densely Connected Face Proposal Network"
    最小脸 40px，30FPS@CPU, 250FPS@GPU。

**目录**

.. toctree::
   :maxdepth: 1
    
   RefineDet
   200FPS
   fast_fd
   SRN
   DSFD


Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
