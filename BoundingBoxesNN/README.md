# Bounding Boxes CNN
These scripts can be used to train a CNN for the task of single object detection.
The CNN leads a probability whether the object is present in the image or not
and estimate of the bounding boxes coordinates. The loss function used for training
is inspired by the [Fast R-CNN](http://ieeexplore.ieee.org/document/7410526/) paper.
The scripts are optimized to train a version of
[VGG16 pretrained on ImageNet](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM).
For our project we gathered our own [data](https://wolke.sumpi.org/index.php/s/8pbTi2kpSiGbn27).
You can find the scripts for labeling and preprocessing the data in the respective
subdirectories.
