# Load packages
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import BB_CNN


# Specify train parameter
num_data = 128
batch_size = 2

num_iter = num_data // batch_size

filename_log = 'val.csv'

image_width = 224
image_height = 224


# Load dataset
dataset = tf.data.TFRecordDataset('val.record')
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()


# Set up log files
log_file = open(filename_log, 'w', 1)
log_file.write('num_data,tpr,tnr,fpr,fnr,mean_bb_err\n')
err_file = open('error.log', 'w', 1)

with tf.Session() as sess:
    # Create network
    bb_net = BB_CNN.BB_CNN(kernel_size = 13 * [3], kernel_stride = 13 * [1],
                           num_filters =  2 * [64] + 2 * [128] + 3 * [256] + 6 * [512],
                           pool_size = 2 * [1, 2] + 3 * [1, 1, 2], pool_stride = 2 * [1, 2] + 3 * [1, 1, 2],
                           hidden_dim = 2 * [4096], dropout = 0.5, weight_decay_bb = 0.0, weight_scale = 1e-3,
                           file_name = 'vgg16.npy', loss_bb_weight = 1.0)
    
    # Build computational graph and calculate loss
    images = tf.placeholder(tf.float32, [batch_size, image_width, image_height, 3])
    train_mode = tf.placeholder(tf.bool)
    target_prob = tf.placeholder(tf.float32, [batch_size])
    target_bb = tf.placeholder(tf.float32, [batch_size, 4])
    bb_net.build(images, train_mode)
    bb_net.predict()
    
    # Build graph for parsing
    # Define features for parsing the TFRecord file
    feature = {'image/encoded': tf.FixedLenFeature([], tf.string),
               'image/object/bbox/xmin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value = -1.),
               'image/object/bbox/xmax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value = 0.),
               'image/object/bbox/ymin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value = -1.),
               'image/object/bbox/ymax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value = 0.)}
    next_example = tf.placeholder(tf.string, [batch_size])
    parser = tf.parse_example(next_example, features=feature)
    
    # Build graph for image decoding
    encoded_images = tf.placeholder(tf.string, [batch_size])
    image_decoder = tf.reverse(tf.map_fn(lambda var: tf.cast(tf.image.decode_jpeg(var), tf.float32), 
                         encoded_images, dtype=tf.float32), [-1]) - tf.constant([[[[103.939, 116.779, 123.68]]]])
    
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    
    bb_err, sum_bb, tp, fp, tn, fn, sum_pred = 7 * [0]
    # Validation Loop
    for i in range(num_iter):
        next_example_val = sess.run(next_element)
        try:
            parsed_out = sess.run(parser, feed_dict={next_example: next_example_val})

            xmin = np.array(parsed_out['image/object/bbox/xmin'])
            xmax = np.array(parsed_out['image/object/bbox/xmax'])
            ymin = np.array(parsed_out['image/object/bbox/ymin'])
            ymax = np.array(parsed_out['image/object/bbox/ymax'])

            if xmin.size == 0:
                xmin = ymin = np.array(batch_size * [[-1.]])
                xmax = ymax = np.array(batch_size * [[0.]])

            prob = np.array(list(map(lambda x: x[0] > -.5, xmin)))

            x = sess.run(image_decoder, feed_dict={encoded_images: parsed_out['image/encoded']})
            net_prob = np.array(sess.run(bb_net.pred_prob, feed_dict={images: x, train_mode: False}) > .5)
            sum_pred += batch_size
            # Count true positive, etc.
            tp += np.sum((prob == net_prob) & prob)
            tn += np.sum((prob == net_prob) & (~prob))
            fp += np.sum((prob != net_prob) & prob)
            fn += np.sum((prob != net_prob) & (~prob))
            net_bb = sess.run(bb_net.pred_bb, feed_dict={images: x, train_mode: False})
            bb_err += np.sum(prob * np.mean(np.abs(np.concatenate((xmin, ymin, xmax - xmin, ymax - ymin), axis = 1) - net_bb), 1))
            sum_bb += np.sum(prob)
            
        except Exception as ex:
            err_file.write(str(ex) + '\n')
                
    if sum_bb == 0:
        mean_bb_err = 0.
    else:
        mean_bb_err = np.sum(bb_err) / sum_bb
        
    if sum_pred == 0:
        tpr, fpr, tnr, fnr = 4 * [0.]
    else:
        tpr, fpr, tnr, fnr = np.array([tp, fp, tn, fn]) / sum_pred
                
    log_file.write(str(sum_pred) + ',' + str(tpr) + ',' + str(tnr) + ',' + str(fpr) + ',' + str(fnr) + ',' + str(mean_bb_err) + '\n')
            
# Close output files
log_file.close()
err_file.close()