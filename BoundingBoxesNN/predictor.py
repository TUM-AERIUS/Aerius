import numpy as np
import tensorflow as tf
import argparse
import re
import cv2
import os
import time

import BB_CNN
from dataset_scripts import resize_images

class Predictor:
    def __init__(self, model, label_map_path, score_threshold=0.5):
        self.model = model
        self._load_label_map(label_map_path)
        # Set defaults
        self.score_threshold = score_threshold
        self.visualize = False
        # Set default methods to work with localization model
        self.predict = self._predict_with_localization_model
        self.initialize_graph = self._initialize_localization_graph


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


    # Set up attributes for usage of a detection API model
    def use_detection_model(self):
        self.initialize_graph = self._initialize_detection_graph
        self.predict = self._predict_with_detection_model

    # Load localization model
    def _initialize_localization_graph(self):
        bb_net = BB_CNN.BB_CNN(kernel_size=13 * [3], kernel_stride=13 * [1],
                               num_filters=2 * [64] + 2 * [128] + 3 * [256] + 6 * [512],
                               pool_size=2 * [1, 2] + 3 * [1, 1, 2], pool_stride=2 * [1, 2] + 3 * [1, 1, 2],
                               hidden_dim=2 * [4096], dropout=0.5, weight_decay_bb=0.0, weight_scale=1e-3,
                               file_name=self.model, loss_bb_weight=1.0)
        images = tf.placeholder(tf.float32, [1, 224, 224, 3])
        bb_net.build(images)
        self.graph = bb_net
        self.graph_input_images = images

    # Predict object bounding boxes with corresponding labels and scores for a given numpy image
    def _predict_with_localization_model(self, image_np):
        # Resize image
        from PIL import Image
        image = Image.fromarray(image_np, 'RGB')
        image, _ = resize_images.resize_pil_image_with_padding(image, 224, 224)
        image = np.array(image)
        # Swap rgb to bgr
        image = np.flip(image, 2)
        # Substract image mean
        mean = np.array([[[103.939, 116.779, 123.68]]])
        image = image-mean
        # Add image into empty array, because predictor expects a batch of images (with size 1)
        input_image = np.array([image])
        b = p = None
        # Run localization model on image
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            (b, p) = self.graph.predict()
            (b, p) = sess.run([b, p], feed_dict={self.graph_input_images: input_image})
        b, p = np.squeeze(b), np.squeeze(p).item(0)
        # change b format from [left, top, widht, height] to [top, left, bot, right]
        b = np.array([b[1], b[0], b[1]+b[3], b[0]+b[2]])
        return np.array([[b, p, 1]])

    # Load frozen detection model
    def _initialize_detection_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    # Predict objects using a model trained with the tensorflow object detection API
    def _predict_with_detection_model(self, image_np):
        with self.graph.as_default():
            with tf.Session(graph=self.graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
                boxes = self.graph.get_tensor_by_name('detection_boxes:0')
                scores = self.graph.get_tensor_by_name('detection_scores:0')
                classes = self.graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                return zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes))


    # Set self._visualize to true and creates the category index that will be needed for plotting
    def activate_visualization(self):
        self.visualize = True

    # Draw given detections onto the given image
    def _draw_detections(self, image, detections):
        image_h, image_w, _ = image.shape
        for (b, s, c) in detections:
            if s >= self.score_threshold:
                b_top, b_bot = int(b[0] * image_h), int(b[2] * image_h)
                b_left, b_right = int(b[1] * image_w), int(b[3] * image_w)
                label = self.label_map_dict[c]
                cv2.rectangle(image, (b_left, b_top), (b_right, b_bot), (255, 0, 0), 2)
                cv2.putText(image, label, (b_left, b_bot), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)


    # Periodically takes images from the primary camera and runs the detection model on them.
    # Detection continues until 'q' is pressed.
    def process_webcam(self):
        cap = cv2.VideoCapture(0)
        while True:
            _, image_np = cap.read()
            detections = self.predict(image_np)
            # TODO: save detections or perform further computation
            if self.visualize:
                self._draw_detections(image_np, detections)
                cv2.imshow("Predictor", image_np)
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
        camera.resolution = (224, 126)
        camera.framerate = 10
        rawCapture = PiRGBArray(camera, size=camera.resolution)
        # warmup camera
        time.sleep(0.1)
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image_np = frame.array
            detections = self.predict(image_np)
            # TODO: save detections or perform further computation
            if self.visualize:
                self._draw_detections(image_np, detections)
                cv2.imshow("Predictor", image_np)
            # Clear the stream in preparation for the next frame and if the 'q' key was pressed, break from the loop
            key = cv2.waitKey(25) & 0xFF
            rawCapture.truncate(0)
            if key == ord("q"):
                break

    # Run detection model on all images in image_dir.
    def process_image_directory(self, image_dir):
        # Set video_path and log dir to default if required
        video_path = os.path.join(os.getcwd(), 'detection_results.avi')
        log_dir = os.path.join(os.getcwd(), 'detection_logs')
        # Create log dir if it does not exist yet
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Create sorted array of all images that should be processed
        images = os.listdir(image_dir)
        image_paths = [os.path.join(image_dir, image) for image in images if
                       os.path.isfile(os.path.join(image_dir, image))]
        image_paths.sort()
        video = None
        for image_num, image_path in enumerate(image_paths):
            # Read images
            imagename = os.path.basename(image_path)
            print('processing image ' + imagename)
            image = cv2.imread(image_path)
            # Perform detection and log time
            time_start = time.time()
            detections = self.predict(image)
            execution_time = time.time() - time_start
            # Write elapsed time, number of detections, boxes, scores and classes to log file
            with open(os.path.join(log_dir, imagename + '.txt'), 'w') as f:
                f.write('Execution Time: ' + str(execution_time)
                        + '\n\nObjects:\n')
                for (b, s, c) in detections:
                    f.write(str(c) + ', '
                            + '[' + str(b[0]) + '-' + str(b[2])
                            + ',' + str(b[1]) + '-' + str(b[3]) + ']\n')
            if self.visualize:
                # Initialize video before first frame
                if image_num == 0:
                    print(' -dCreating video...')
                    height, width, _ = image.shape
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
                # Draw detections on image and write to video
                self._draw_detections(image, detections)
                video.write(image)
                # Close video after the last frame
                if image_num == len(image_paths) - 1:
                    print('Video released!')
                    cv2.destroyAllWindows()
                    video.release()


def create_parser():
    parser = argparse.ArgumentParser(description='run frozen tensorflow model on webcam images')
    parser.add_argument('model',
                        help="Path to frozen tensorflow model (.pb)")
    parser.add_argument('label_map',
                        help="Path to label_map (.pbtxt)")
    parser.add_argument('--use_detection_model', '-d', dest='det_model', action='store_true',
                        help="Set this flag when using a model trained with the detection API")
    parser.add_argument('--use_picam', '-p', dest='picam', action='store_true',
                        help="Set this flag when using picam of raspberry pi instead of regular webcams")
    parser.add_argument('--show', '-s', dest='visualize', action='store_true',
                        help="Visualize camera detection results")
    parser.add_argument('--use_image_directory', '-i',
                        help="Provide this flag together with an image directory to run the predictor on all images "
                             "in the directory, instad on live camera images. When chosing this options, detection "
                             "results are logged to log files. If the -s flag is also set, a video of the detection "
                             "results will be created, instead of plotting them directly.")
    parser.set_defaults(det_model=False, picam=False, visualize=False)
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    predictor = Predictor(args.model, args.label_map)
    image_dir = args.use_image_directory
    # Determine model type and whether results should be plotted
    if args.det_model:
        predictor.use_detection_model()
    if args.visualize:
        predictor.activate_visualization()
    # initialize detection graph
    predictor.initialize_graph()
    # Determine processing mode
    if image_dir:
        predictor.process_image_directory(image_dir)
    elif args.picam:
        predictor.process_picam()
    else:
        predictor.process_webcam()

if __name__ == '__main__':
    main()
