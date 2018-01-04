import numpy as np
import tensorflow as tf
import argparse
import re
import cv2


class Predictor:
    def __init__(self, detection_model, label_map_path):
        self.detection_model = detection_model
        self._load_label_map(label_map_path)
        self._initialize_graph()

    # Loads the label map specified in label_map_path, transforms it to a (id->display_name) dictionary and assigns it to self.label_map_dict
    def _load_label_map(self, label_map):
        label_map_dict = {}
        re_id = r'id: (\d*)'
        re_dn = r'name: (.*)'
        re_item_end = r'.*}.*'
        with open(label_map, 'r') as f:
            id = -1
            display_name = ''
            for line in f:
                if re.search(re_id,line):
                    id = float(re.search(re_id,line).group(1))
                if re.search(re_dn,line):
                    display_name =  re.search(re_dn,line).group(1)
                if re.search(re_item_end, line):
                    label_map_dict[id] = display_name
        self.label_map_dict = label_map_dict
        self.label_map_path = label_map

    # Creates a category index, which will be used to translate detection names when plotting
    def _create_category_index(self):
        from utils import label_map_util
        label_map = self.label_map_path
        num_classes = 0
        with open(label_map, 'r') as f_in:
            for line in f_in:
                if re.match(r".*id: (\d*).*", line):
                    num_classes += 1
        label_map_temp = label_map_util.load_labelmap(label_map)
        categories = label_map_util.convert_label_map_to_categories(label_map_temp, max_num_classes=num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    # Load frozen detection model
    def _initialize_graph(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.detection_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    # Predict object bounding boxes with corresponding labels and scores for a given numpy image
    def predict(self, image_np):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                return (boxes, scores, classes)

    # Set self._visualize to true and creates the category index that will be needed for plotting
    def activate_visualization(self):
        self._visualize = True
        self._create_category_index()

    # Visualize the image with given detection results
    def _visualize_image(self, image_np, boxes, scores, classes):
        from utils import visualization_utils as vis_util
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

    # Periodically takes images from the primary camera and runs the detection model on them.
    # Detection continues until 'q' is pressed.
    def process_camera(self):
        cap = cv2.VideoCapture(0)
        while True:
            _, image_np = cap.read()
            (boxes, scores, classes) = self.predict(image_np)
            labels = [self.label_map_dict[label] for label in np.squeeze(classes)]
            # TODO: save detections or perform further computation
            if self._visualize:
                self._visualize_image(image_np, boxes, scores, classes)
            # if the 'q' key was pressed, break from the loop
            key = cv2.waitKey(25) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                break

    # Runs the detection model on camera images.
    # In difference to process_camera(), the images are taken from a picam.
    def process_picam(self):
        # import here to prevent errors
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        import time
        camera = PiCamera()
        camera.resolution = (1280, 720)
        camera.framerate = 32
        rawCapture = PiRGBArray(camera, size=(1280, 720))
        # warmup camera
        time.sleep(0.1)
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image_np = frame.array
            (boxes, scores, classes) = self.predict(image_np)
            # TODO: save detections or perform further computation
            if self._visualize:
                self._visualize_image(image_np, boxes, scores, classes)
            # Clear the stream in preparation for the next frame and if the 'q' key was pressed, break from the loop
            key = cv2.waitKey(25) & 0xFF
            rawCapture.truncate(0)
            if key == ord("q"):
                break


def create_parser():
    parser = argparse.ArgumentParser(description='run frozen tensorflow model on webcam images')
    parser.add_argument('model',
                        help="Path to frozen tensorflow model (.pb)")
    parser.add_argument('label_map',
                        help="Path to label_map (.pbtxt)")
    parser.add_argument('--use_picam', '-p', dest='picam', action='store_true',
                        help="Set this flag when using picam of raspberry pi instead of regular webcams")
    parser.add_argument('--show', '-s', dest='visualize', action='store_true',
                        help="Visualize detection results (Requires packages from tensorflow object detection)")
    parser.set_defaults(picam=False, visualize=False)
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    predictor = Predictor(args.model, args.label_map)
    if args.visualize:
        predictor.activate_visualization()
    if args.picam:
        predictor.process_picam()
    else:
        predictor.process_camera()

if __name__ == '__main__':
    main()
