# PyMDNet

by [HungWei-Andy](https://github.com/HungWei-Andy) @ [NTU DISPLab](http://disp.ee.ntu.edu.tw/)

## Introduction
Python (tensorflow) implementation of **[Hyeonseob Nam](https://kr.linkedin.com/in/hyeonseob-nam/) and [Bohyung Han](http://cvlab.postech.ac.kr/%7Ebhhan/), [Learning Multi-Domain Convolutional Neural Networks.](https://arxiv.org/abs/1510.07945), CVPR2016**. 

## Prequisites
### Python Version
Python 2.7

### Requirements
numpy>=1.12.1
tensorflow-gpu==1.0.0
matplotlib>=2.0.1
skimage>=0.13.0
Pillow>=2.2.1

## Usage

### Data Directory
Please download and put the video files of [OTB](http://cvlab.hanyang.ac.kr/tracker_benchmark/benchmark_v10.html) and [VOT](http://www.votchallenge.net/) in the directory 'data' with structure as follows:
++ data
+++ otb
++++ Basketball
...

+++ vot
++++ vot2013
+++++ bicycle
...

++++ vot2014
+++++ ball
...

++++ vot2015
+++++ bag
...


### Tracking
```bash
  python tracking.py --dataset dataset --seq sequence --load_path path [--no_display]
```
 
### Pretraining
``` bash
  bash pretrain.sh
```

