import argparse
import subprocess
import shutil
import os
import hashlib
import io
import logging
import re
import sys

from functools import reduce

from lxml import etree
from PIL import Image
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2" # disable tf debug logging

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

class Converter:

    def __init__(self):
        self._REGEX_IMAGE_END = r".*</image>.*"
        self._REGEX_IMAGE_START = r".*<image file='(.*\.\w*)'>.*"
        self._REGEX_BOX_START = r".*<box top='\d*' left='\d*' width='\d*' height='\d*'>.*"
        self._REGEX_TOP = r"top='(-?\d*)'"
        self._REGEX_LEFT = r"left='(-?\d*)'"
        self._REGEX_WIDTH = r"width='(-?\d*)'"
        self._REGEX_HEIGHT = r"height='(-?\d*)'"
        self._REGEX_LABEL = r".*<label>(.*)</label>.*"
        self._REGEX_BOX_END = r".*</box>.*"

    # Split a given path into directory, filename and extension
    def _split_path_into_dir_filename_and_extension(self, path):
        directory = os.path.dirname(path)
        base_parts = os.path.splitext(os.path.basename(path))
        # the filename is everything except for the last base_part
        # if the filename does not contain '.', the following is equivalent to 'filename = base_parts[0]'
        filename = reduce(lambda x, y: str(x)+"."+str(y), base_parts[1:-1], base_parts[0])
        ext = base_parts[-1]
        return directory, filename, ext


    # Convert all images and annotations corresponding to input_imglab_file to jpg and save the result to output_imglab_file.
    # Returns a list of paths to all created images
    def png_to_jpg(self, input_imglab_file, output_imglab_file):
        created_images = []
        with open(input_imglab_file, 'r') as f_in:
            with open(output_imglab_file, 'w+') as f_out:
                for line in f_in:
                    # Change png images to jpg
                    if re.match(self._REGEX_IMAGE_START, line) and re.search(self._REGEX_IMAGE_START, line).group(1).endswith(".png"):
                        # Convert image
                        filepath = re.search(self._REGEX_IMAGE_START, line).group(1)
                        new_filepath = filepath[:-4] + ".jpg"
                        with Image.open(filepath) as img:
                            img.convert('RGB').save(new_filepath)
                            created_images.append(new_filepath)
                        # Change filename in imglab file
                        f_out.write("\t" + r"<image file='" + new_filepath + r"'>" + "\n")
                    else:
                        f_out.write(line)
        return created_images


    # Converts given imglab file to Pascal VOC
    # Returns path to the created imglab annotation directory (dir_out, if set, else <dir>/annotations_VOC/)
    def convert_imglab_to_pascalvoc(self, imglab_file, dir_out=None):

        dir_img, imglab_filename, _ = self._split_path_into_dir_filename_and_extension(imglab_file)

        # If output directory is not provided, set to default (input_dir/annotations_VOC/)
        dir_out = dir_out if dir_out else os.path.join(dir_img, 'annotations_VOC')

        # Create output directories
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        dir_xml = os.path.join(dir_out, "xmls")
        if not os.path.exists(dir_xml):
            os.makedirs(dir_xml)

        # Initialize variables
        l, t, r, b, img_w, img_h = [0] * 6
        filepath = None
        line_list = []
        in_image = False
        in_box = False

        # Perform some spaghetti magic   (>^.^)> **~~**~~** ======================================== **~~**~~** <(^.^<)
        with open(os.path.join(dir_out, 'examples.txt'), 'w+') as f_examples:
            with open(imglab_file, 'r') as f_in:
                for line in f_in:
                    if not in_image and re.match(self._REGEX_IMAGE_START, line):
                        in_image = True
                        filepath = (re.search(self._REGEX_IMAGE_START, line).group(1))
                        with Image.open(dir_img + filepath) as img:
                            img_w, img_h = img.size
                    elif in_image:
                        if re.match(self._REGEX_BOX_START, line):
                            in_box = True
                            t = int(re.search(self._REGEX_TOP, line).group(1))
                            l = int(re.search(self._REGEX_LEFT, line).group(1))
                            w = int(re.search(self._REGEX_WIDTH, line).group(1))
                            h = int(re.search(self._REGEX_HEIGHT, line).group(1))
                            r = l + w
                            b = t + h
                        if in_box and re.match(self._REGEX_LABEL, line):
                            label = re.search(self._REGEX_LABEL, line).group(1)
                            line_list.append("\t<object>\n")
                            line_list.append("\t\t<name>" + str(label) + "</name>\n")
                            line_list.append("\t\t<pose>unknown</pose>\n")
                            line_list.append("\t\t<truncated>0</truncated>\n")
                            line_list.append("\t\t<occluded>0</occluded>\n")
                            line_list.append("\t\t<bndbox>\n")
                            line_list.append("\t\t\t<xmin>" + str(l) + "</xmin>\n")
                            line_list.append("\t\t\t<ymin>" + str(t) + "</ymin>\n")
                            line_list.append("\t\t\t<xmax>" + str(r) + "</xmax>\n")
                            line_list.append("\t\t\t<ymax>" + str(b) + "</ymax>\n")
                            line_list.append("\t\t</bndbox>\n")
                            line_list.append("\t\t<difficult>0</difficult>\n")
                            line_list.append("\t</object>\n")
                        if in_box and re.match(self._REGEX_BOX_END, line):
                            in_box = False
                        elif re.match(self._REGEX_IMAGE_END, line):
                            in_image = False
                            filename = os.path.basename(filepath)
                            f_examples.write(filename + "\n")
                            with open(os.path.join(dir_xml, filename + ".xml"), 'w+') as f_out:
                                f_out.write("<annotation>\n"
                                            "\t<folder>unknown</folder>\n"
                                            "\t<filename>" + filepath + "</filename>\n"
                                            "\t<source>\n"
                                            "\t\t<database>unknown</database>\n"
                                            "\t\t<annotation>unknown</annotation>\n"
                                            "\t\t<image>unknown</image>\n"
                                            "\t</source>\n"
                                            "\t<size>\n"
                                            "\t\t<width>" + str(img_w) + "</width>\n"
                                            "\t\t<height>" + str(img_h) + "</height>\n"
                                            "\t\t<depth>3</depth>\n"
                                            "\t</size>\n"
                                            "\t<segmented>0</segmented>\n")
                                for line_list_line in line_list:
                                    f_out.write(line_list_line)
                                line_list = []
                                f_out.write("</annotation>")
        # End of spaghetti magic. You have seen some shit, haven't you? ???(>o__O)>???
        return dir_out


    # Write one entry for each given label (labels) to the label map file (out)
    def _write_labels_to_label_map_file(self, labels, out):
        with open(out, 'w') as f_out:
            for i in range(len(labels)):
                f_out.write("item {\n\tid: " + str(i + 1) + "\n\tname: '" + labels[i] + "'\n}\n")

    # Create a label map (.pbtxt) for the given imglab annotation file
    # Returns path to the created label map
    def create_label_map(self, imglab_file, out=None):
        # Get all labels that occur in the given imglab file
        labels_string = subprocess.check_output(['imglab', '-l', imglab_file])
        labels = labels_string.split()
        # Sets out to default path, if it was not defined
        if not out:
            file_dir, filename, _ = self._split_path_into_dir_filename_and_extension(imglab_file)
            out = os.path.join(file_dir, filename + '_label_map.pbtxt')
        # Write the label map
        self._write_labels_to_label_map_file(labels, out)
        return out


    # The following two helper functions perform the actual conversion to TFRecord
    # The code is mostly CTRL+C/CTRL+V from the original example in the TF Object Detection github   \_(^u^)_/
    # (models/research/object_detection/dataset_tools/create_pascal_tf_record.py)
    def _dict_to_tf_example(self, data, label_map_dict, image_subdirectory, ignore_difficult_instances=False):
        """Convert XML derived dict to tf.Example proto.
        Notice that this function normalizes the bounding box coordinates provided
        by the raw data.
        Args:
          data: dict holding PASCAL XML fields for a single image (obtained by
            running dataset_util.recursive_parse_xml_to_dict)
          label_map_dict: A map from string label names to integers ids.
          image_subdirectory: String specifying subdirectory within the
            Pascal dataset directory holding the actual image data.
          ignore_difficult_instances: Whether to skip difficult instances in the
            dataset  (default: False).
        Returns:
          example: The converted tf.Example.
        Raises:
          ValueError: if the image pointed to by data['filename'] is not a valid JPEG
        """
        img_path = os.path.join(image_subdirectory, data['filename'])
        with tf.gfile.GFile(img_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()
        width = int(data['size']['width'])
        height = int(data['size']['height'])
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []
        if 'object' in data:
            for obj in data['object']:
                difficult = bool(int(obj['difficult']))
                if ignore_difficult_instances and difficult:
                    continue
                difficult_obj.append(int(difficult))
                xmin.append(float(obj['bndbox']['xmin']) / width)
                ymin.append(float(obj['bndbox']['ymin']) / height)
                xmax.append(float(obj['bndbox']['xmax']) / width)
                ymax.append(float(obj['bndbox']['ymax']) / height)
                # class_name = get_class_name_from_filename(data['filename'])
                class_name = obj['name']
                classes_text.append(class_name.encode('utf8'))
                classes.append(label_map_dict[class_name])
                truncated.append(int(obj['truncated']))
                poses.append(obj['pose'].encode('utf8'))
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
        }))
        return example
    def _create_tf_record(self, output_filename, label_map_dict, annotations_dir, image_dir, examples):
        """Creates a TFRecord file from examples.
        Args:
          output_filename: Path to where output file is saved.
          label_map_dict: The label map dictionary.
          annotations_dir: Directory where annotation files are stored.
          image_dir: Directory where image files are stored.
          examples: Examples to parse and save to tf record.
        """
        writer = tf.python_io.TFRecordWriter(output_filename)
        for idx, example in enumerate(examples):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples))
            path = os.path.join(annotations_dir, 'xmls', example + '.xml')
            if not os.path.exists(path):
                logging.warning('Could not find %s, ignoring example.', path)
                continue
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
            tf_example = self._dict_to_tf_example(data, label_map_dict, image_dir)
            writer.write(tf_example.SerializeToString())
        writer.close()

    # Converts given images (image_dir) and corresponding Pascal VOC annotations (pascal_voc_dir) to TFRecord (out)
    # To do so, a label map (label_map), as well as an example list (examples) are required
    # Returns path to the created TFRecord
    def convert_pascalvoc_to_tfrecord(self, image_dir, pascal_voc_dir, label_map, output_filename, examples):
        label_map_dict = label_map_util.get_label_map_dict(label_map)
        examples = dataset_util.read_examples_list(examples)
        self._create_tf_record(output_filename, label_map_dict, pascal_voc_dir, image_dir, examples)


    # Convert a given imglab annotation file (.xml) to TFRecord (.record)
    # Returns path to the created TFRecord
    def convert_imglab_to_tfrecord(self, imglab_file, tfrecord_out=None, label_map_out=None, rmempty=False):

        # If the TFRecord output path is not provided, set to default
        dir_img, imglab_filename, _ = self._split_path_into_dir_filename_and_extension(imglab_file)
        # If output directory is not provided, set to default (input_dir/annotations_VOC/)
        tfrecord_out = tfrecord_out if tfrecord_out else os.path.join(dir_img, imglab_filename + '.record')

        # Save a backup of the imglab file (which will be restored at the end) and convert all .pngs to jpg
        imglab_file_backup = imglab_file + '.i_t_tfr_BACKUP'
        print('Converting .png images to .jpg...')
        subprocess.call(["mv", imglab_file, imglab_file_backup])
        created_images = self.png_to_jpg(imglab_file_backup, imglab_file)

        # If rmempty is set, remove empty images from the imglab_file
        if rmempty:
            print('Removing empty images...')
            subprocess.call(["imglab", "--rmempty", imglab_file])
            subprocess.call(["mv", imglab_file+".rmempty.xml", imglab_file])

        # Convert from Imglab to Pascal VOC first
        print('Converting from Imglab to Pascal VOC...')
        pascal_voc_dir = self.convert_imglab_to_pascalvoc(imglab_file)

        # Create labelmap
        print('Creating label map...')
        label_map_out = self.create_label_map(imglab_file, out=label_map_out)

        # Convert from Pascal VOC to TFRecord, using the created labelmap
        print('Converting from Pascal VOC to TFRecord...')
        examples = os.path.join(pascal_voc_dir, "examples.txt")
        tfrecord_out = self.convert_pascalvoc_to_tfrecord(dir_img, pascal_voc_dir, label_map_out, tfrecord_out, examples)

        # Cleanup pascal voc files and, if rmempty was set, restore original imglab file
        print('Cleaning up...')
        shutil.rmtree(os.path.join(os.getcwd(), pascal_voc_dir))
        for created_image in created_images:
            os.remove(created_image)
        subprocess.call(["mv", imglab_file_backup, imglab_file])

        print('done!')
        return tfrecord_out



# Create argument parser
def create_parser():
    parser = argparse.ArgumentParser(description='Convert imglab annotation file(s) (.xml) to TFRecord (.record)')
    parser.add_argument('imglab_files', nargs='+', help='Path(s) to imglab annotation file(s) (.xml)')
    parser.add_argument('--outputs', '-o', nargs='*',
                        help='Path(s) to output TFRecord(s) (.xml). Default: <input_filename>.record')
    parser.add_argument('--label_map_outputs', '-l',
                        help='Path(s) to output label map(s). Default: <input_filename>_label_map.pbtxt')
    parser.add_argument('--rmempty', '-r', dest='rmempty', action='store_true',
                        help='Remove all images without annotation during conversion')
    parser.set_defaults(rmempty=False)
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    converter = Converter()

    imglab_files = args.imglab_files
    output_files = args.outputs
    label_maps = args.label_map_outputs
    rmempty = args.rmempty

    # Exit if -o or -l was given but number of corresponding arguments does not match number of inputs
    if output_files and len(output_files) != len(imglab_files):
        sys.exit("InvalidArgumentException: -o flag was given, but number of outputs does not match number of inputs. "
                 "Please provide one output path for each input file.")
    if label_maps and len(label_maps) != len(imglab_files):
        sys.exit("InvalidArgumentException: -l flag was given, but number of label map pahts does not match number of "
                 "inputs. Please provide one label map path for each input file.")

    label_maps = [None]*len(imglab_files) if label_maps is None else label_maps
    output_files = [None]*len(imglab_files) if output_files is None else output_files

    for i, imglab_file in enumerate(imglab_files):
        print("STARTING CONVERTION OF " + imglab_file + ":")
        converter.convert_imglab_to_tfrecord(imglab_file, output_files[i], label_maps[i], rmempty)

if __name__ == '__main__':
    main()
