import tensorflow as tf

batch_size = 100
keep_prob = 0.8

def neural_net(num_sensors, params):

    # setting up the environment

    tf.reset_default_graph()

    # building the graph

    input = tf.placeholder(shape=[batch_size, num_sensors], dtype=tf.float32)
    W1 = tf.Variable(tf.random_uniform([num_sensors, params[0]], 0, 0.01), dtype=tf.float32)
    layer1 = tf.nn.relu(tf.matmul(input, W1))
    layer1 = tf.nn.dropout(layer1, keep_prob)
    W2 = tf.Variable(tf.random_uniform([params[0], params[1]], 0, 0.01), dtype=tf.float32)
    layer2 = tf.nn.relu(tf.matmul(layer1, W2))
    layer2 = tf.nn.dropout(layer2, keep_prob)
    W3 = tf.Variable(tf.random_uniform([params[1], 3], 0, 0.01), dtype=tf.float32)
    logits = tf.nn.relu(tf.matmul(layer2, W3))

    # calculating loss

    labels = tf.placeholder(shape=[batch_size, 3], dtype=tf.float32)
    # labels = tf.nn.sigmoid(labels)
    # explicit declaration of the loss function
    loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(labels - logits))
    )
    # loss = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    # )

    # training

    trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
    update = trainer.minimize(loss)

    # prediction

    state = tf.placeholder(shape=[None, num_sensors], dtype=tf.float32)
    player1 = tf.nn.relu(tf.matmul(state, W1))
    player2 = tf.nn.relu(tf.matmul(player1, W2))
    prediction = tf.nn.sigmoid(tf.matmul(player2, W3))

    # initialize session

    init = tf.global_variables_initializer()

    session = tf.Session()

    session.run(init)

    return session, update, prediction, state, input, labels
