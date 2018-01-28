# Load packages
import numpy as np
import tensorflow as tf
import BB_CNN


# Specify train parameter
epochs = 1
num_train = 6
batch_size = 2
log_every = 1
num_val = 1
save_every = num_train // batch_size

num_iter = epochs * (num_train // batch_size)

starter_learning_rate = 1e-5
learning_decay_rate = 0.65
use_dropout = False

filename_log = 'log.csv'

image_width = 224
image_height = 224


# Load train dataset
dataset_train = tf.data.TFRecordDataset('train.record')
dataset_train = dataset_train.concatenate(tf.data.TFRecordDataset('train_flipped.record'))
dataset_train = dataset_train.shuffle(buffer_size=10000)
dataset_train = dataset_train.batch(batch_size)
dataset_train = dataset_train.repeat()
iterator_train = dataset_train.make_initializable_iterator()
next_element_train = iterator_train.get_next()

# Load validation dataset
dataset_val = tf.data.TFRecordDataset('test.record')
dataset_val = dataset_val.concatenate(tf.data.TFRecordDataset('test_flipped.record'))
dataset_val = dataset_val.concatenate(tf.data.TFRecordDataset('val.record'))
dataset_val = dataset_val.concatenate(tf.data.TFRecordDataset('val_flipped.record'))
dataset_val = dataset_val.shuffle(buffer_size=4000)
dataset_val = dataset_val.batch(batch_size)
dataset_val = dataset_val.repeat()
iterator_val = dataset_val.make_initializable_iterator()
next_element_val = iterator_val.get_next()


# Set up log files
log_file = open(filename_log, 'w', 1)
log_file.write('iteration,train loss,train pred acc,train bb mean abs err,val loss,val pred acc,val bb mean abs err\n')
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
    bb_net.loss(target_prob, target_bb)
    
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
    
    # Declare optimizer
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, num_train // batch_size, learning_decay_rate, staircase=False)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(bb_net.loss, global_step=global_step)
    
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(iterator_train.initializer)
    sess.run(iterator_val.initializer)
    
    # Train Loop
    for i in range(num_iter):
        next_example_train = sess.run(next_element_train)
        try:
            parsed_out = sess.run(parser, feed_dict={next_example: next_example_train})
            
            xmin = np.array(parsed_out['image/object/bbox/xmin'])
            xmax = np.array(parsed_out['image/object/bbox/xmax'])
            ymin = np.array(parsed_out['image/object/bbox/ymin'])
            ymax = np.array(parsed_out['image/object/bbox/ymax'])
            
            if xmin.size == 0:
                xmin = ymin = np.array(batch_size * [[-1.]])
                xmax = ymax = np.array(batch_size * [[0.]])
            
            prob = list(map(lambda x: float(x[0] > -.5), xmin))
            x = sess.run(image_decoder, feed_dict={encoded_images: parsed_out['image/encoded']})
            sess.run(train_step, feed_dict={images: x, train_mode: use_dropout, target_prob: prob, 
                                            target_bb: np.concatenate((xmin, ymin, np.log(xmax - xmin), np.log(ymax - ymin)), axis = 1)})
        except Exception as ex:
            err_file.write(str(ex) + '\n')
        
        # Log of learning progress (loss, acc, bb mean abs err)
        if (i + 1) % log_every == 0:
            loss_train = np.zeros(num_val)
            acc_train = np.zeros(num_val)
            bb_err_train = np.zeros(num_val)
            loss_val = np.zeros(num_val)
            acc_val = np.zeros(num_val)
            bb_err_val = np.zeros(num_val)
            sum_obj_train = 0
            sum_obj_val = 0
            for j in range(num_val):
                next_example_val = sess.run(next_element_train)
                try:
                    parsed_out = sess.run(parser, feed_dict={next_example: next_example_val})

                    xmin = np.array(parsed_out['image/object/bbox/xmin'])
                    xmax = np.array(parsed_out['image/object/bbox/xmax'])
                    ymin = np.array(parsed_out['image/object/bbox/ymin'])
                    ymax = np.array(parsed_out['image/object/bbox/ymax'])

                    if xmin.size == 0:
                        xmin = ymin = np.array(batch_size * [[-1.]])
                        xmax = ymax = np.array(batch_size * [[0.]])

                    prob = np.array(list(map(lambda x: float(x[0] > -.5), xmin)))

                    x = sess.run(image_decoder, feed_dict={encoded_images: parsed_out['image/encoded']})
                    net_prob = sess.run(bb_net.pred_prob, feed_dict={images: x, train_mode: False})
                    acc_train[j] = np.mean(np.abs(prob - 1. * (net_prob < .5)))
                    net_bb = sess.run(bb_net.pred_bb, feed_dict={images: x, train_mode: False})
                    bb_err_train[j] = np.sum(prob * np.mean(np.abs(np.concatenate((xmin, ymin, xmax - xmin, ymax - ymin), axis = 1) - net_bb), 1))
                    sum_obj_train += np.sum(prob)
                    loss_train[j] = sess.run(bb_net.loss, feed_dict={images: x, train_mode: False, target_prob: prob, 
                                                                       target_bb: np.concatenate((xmin, ymin, np.log(xmax - xmin), np.log(ymax - ymin)), axis = 1)})
                except Exception as ex:
                    err_file.write(str(ex) + '\n')
                
                next_example_val = sess.run(next_element_val)
                try:
                    parsed_out = sess.run(parser, feed_dict={next_example: next_example_val})

                    xmin = np.array(parsed_out['image/object/bbox/xmin'])
                    xmax = np.array(parsed_out['image/object/bbox/xmax'])
                    ymin = np.array(parsed_out['image/object/bbox/ymin'])
                    ymax = np.array(parsed_out['image/object/bbox/ymax'])

                    if xmin.size == 0:
                        xmin = ymin = np.array(batch_size * [[-1.]])
                        xmax = ymax = np.array(batch_size * [[0.]])

                    prob = np.array(list(map(lambda x: float(x[0] > -.5), xmin)))

                    x = sess.run(image_decoder, feed_dict={encoded_images: parsed_out['image/encoded']})
                    net_prob = sess.run(bb_net.pred_prob, feed_dict={images: x, train_mode: False})
                    acc_val[j] = np.mean(np.abs(prob - 1. * (net_prob < .5)))
                    net_bb = sess.run(bb_net.pred_bb, feed_dict={images: x, train_mode: False})
                    bb_err_val[j] = np.sum(prob * np.mean(np.abs(np.concatenate((xmin, ymin, xmax - xmin, ymax - ymin), axis = 1) - net_bb), 1))
                    sum_obj_val += np.sum(prob)
                    loss_val[j] = sess.run(bb_net.loss, feed_dict={images: x, train_mode: False, target_prob: prob, 
                                                                       target_bb: np.concatenate((xmin, ymin, np.log(xmax - xmin), np.log(ymax - ymin)), axis = 1)})
                except Exception as ex:
                    err_file.write(str(ex) + '\n')
                    
            if sum_obj_train == 0:
                mean_bb_err_train = 0.
            else:
                mean_bb_err_train = np.sum(bb_err_train) / sum_obj_train
            if sum_obj_val == 0:
                mean_bb_err_val = 0.
            else:
                mean_bb_err_val = np.sum(bb_err_val) / sum_obj_val
                
            log_file.write(str(i + 1) + ',' + str(np.mean(loss_train)) + ',' + str(np.mean(acc_train)) + ',' + str(mean_bb_err_train) + ',' + str(np.mean(loss_val)) + ',' + str(np.mean(acc_val)) + ',' + str(mean_bb_err_val) + '\n')
        
        # Save trained model
        if (i + 1) % save_every == 0:
            bb_net.save(sess, './bb_cnn_vgg16_' + ('%02i' % ((i + 1) // save_every)) + '.npy')
            
# Close output files
log_file.close()
err_file.close()