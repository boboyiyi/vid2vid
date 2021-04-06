# 笔记

## 配置环境

创建v2v conda环境并启动
```
$ conda create -n v2v python=3.6
$ conda activate v2v
```

安装必要的包
```
$ pip install dominate requests
$ pip install dlib
$ pip install numpy==1.16.2
$ pip install scipy
$ pip install scikit-image
$ pip install opencv-python
$ pip install torch==1.2.0 torchvision==0.4.0
```

运行人脸demo
```
$ python scripts/download_datasets.py
$ python scripts/face/download_models.py
$ bash ./scripts/face/test_512.sh
```

生成的人脸位于：./results/edge2face_512/test_latest/

## 训练人脸模型

首先，进行人脸检测
```
$ python data/face_landmark_detection.py train
```

该工程采用dlib进行人脸检测，dlib 68点的人脸关键点定义如下：
![dlib 68人脸特征点](imgs/dlib-68.png "dlib 68人脸特征点")

我添加了在人脸绘制特征点的代码，生成的带关键点的人脸位于：./datasets/face/train_debug/
![dlib 68人脸特征点](imgs/00027.jpg "人脸检测&关键点")

