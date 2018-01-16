# Dataset Scripts
This folder contains scripts to manipulate and convert Imglab annotation files.

## Files
- **`extract_images.sh`**: Extracts png images from .h264. The result is a single folder containing all images, as well as
 a training, validation and test annotation file. Before the script is executed, all .h264 files and all corresponding 
 annotations (imageX.xml) have to be located in the current folder. Which images will belong to train, validation or 
 test can be adjusted with the two script arguments. The first one determines where to split train and validation (i.e. 15)
 and the second one where to split validation and test (i.e. 18). Since we only have 20 videos, setting both arguments
 to 20 or higher will result in no split. During the extraction, all labels are renamed to "trousers" (which is 
 necessary for annotations created with TracknLabel) and image paths are adjusted to the new file naming convention.

- **`png_to_jpg.sh`**: Converts all images in the current folder from png to jpg. Also changes all imglab annotation files provided as arguments accordingly.

- **`remove_invalid_values.py`**: Changes all bounding boxes of the given imglab annotation file, which have boundaries that lie outside of the corresponding image.

- **`imglab_to_tfrecord.py`**: Converts the given imglab annotation file to tfrecord. In order to do so, a label map is first created and the annotations are converted to *Pascal VOC* format, before using the label map to convert images and *Pascal VOC* annotations to *TFRecord*

**Note:** *imglab_to_tfrecord.py* requires packages from the tensorflow object detection api, which can be obtained using
 the following command: <br>
`git clone https://github.com/tensorflow/models/archive/master.zip`

## Conversion Pipeline
In order to obtain the current training, validation and test TFRecords, the following steps were taken 
(paths to scripts should be adjusted accordingly or coppied into the current folder):
1. Download all 20 videos (.h264) and corresponding annotations (.xml) into a single folder
2. In this folder, run `./extract_images.sh 15 18` (15 and 18 are were the dataset will be split (i.e.: train=1-15, val=16-18, test=19-20)). This will result in a new folder called *merged*.
3. Go into merged (`cd merged`)
4. Convert images to jpg via `./png_to_jpg.sh train.xml val.xml test.xml`. This also adjusts the annotations accordingly.
5. Remove invalid bounding box coordinates from all annotations:
    - `python remove_invalid_values.py train.xml`
    - `python remove_invalid_values.py val.xml`
    - `python remove_invalid_values.py test.xml`
6. Convert all annotations with corresponding images to TFRecord:
    - `python imglab_to_tfrecord.py train.xml`
    - `python imglab_to_tfrecord.py val.xml`
    - `python imglab_to_tfrecord.py test.xml`
