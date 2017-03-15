import carmunk
import numpy as np
import random
from nn import neural_net
import timeit
import tensorflow as tf
import os

NUM_INPUT = 46
GAMMA = 0.9  # Forgetting.
save_dir = "tensorboard"
load_dir = "tensorData"

def train_net(session, update, predict, state, input, labels, params):
    """
    Function
    :param session: tensorflow session
    :param update: backpropogation, feed_dict: input
    :param predict: feedforward, feed_dict: state
    :param state: current state
    :param input: minibatch of sars (state, action, reward, new state)
    :param labels: training labels placeholder
    :param params: dictionary: batchsize, buffer, nn
    :return:
    """

    observe = 1000  # Number of frames to observe before training.
    epsilon = 1
    train_frames = 1000000  # Number of frames to play.
    batchSize = params['batchSize']
    buffer = params['buffer']

    saver = tf.train.Saver()
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(save_dir, session.graph)

    if len(os.listdir(load_dir)) > 1:
        saver.restore(session, tf.train.latest_checkpoint(load_dir))

    # summary_writer = tf.summary.FileWriter(save_dir, session.graph)

    # Just stuff used below.
    max_car_distance = 0
    car_distance = 0
    t = 0
    data_collect = []
    replay = []  # stores tuples of (S, A, R, S').

    # Create a new game instance.
    game_state = carmunk.GameState()

    # Get initial state by doing nothing and getting the state.
    _, gameState = game_state.frame_step((2))

    # Let's time it.
    start_time = timeit.default_timer()

    # Run the frames.
    while t < train_frames:

        t += 1
        car_distance += 1

        # Choose an action.
        if random.random() < epsilon or t < observe:
            action = np.random.randint(0, 3)  # random
        else:
            feed_dict = {
                state: gameState
            }
            # Get Q values for each action.
            qval = session.run([predict], feed_dict=feed_dict)
            action = np.argmax(qval)  # best
            print(action)

        # Take action, observe new state and get our treat.
        reward, new_state = game_state.frame_step(action)

        # Experience replay storage.
        replay.append((gameState, action, reward, new_state))

        # If we're done observing, start training.
        if t > observe:

            # If we've stored enough in our buffer, pop the oldest.
            if len(replay) > buffer:
                replay.pop(0)

            # Randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)

            # Get training values.
            X_train, y_train = process_minibatch(session, minibatch, predict, state)

            # Train the model on this batch.
            feed_dict = {
                input: X_train,
                labels: y_train
            }
            session.run([update], feed_dict=feed_dict)

            # Save summaries every 100 frames
            if t % 100 == 0:
                print('Step %d: saving loss' % (t))
                summary_str = session.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, t)
                summary_writer.flush()

        # Update the starting state with S'.
        gameState = new_state

        # Decrement epsilon over time.
        if epsilon > 0.1 and t > observe:
            epsilon -= (1/train_frames)

        # We died, so update stuff.
        if reward == -500:
            # Log the car's distance at this T.
            data_collect.append([t, car_distance])

            # Update max.
            if car_distance > max_car_distance:
                max_car_distance = car_distance

            # Time it.
            tot_time = timeit.default_timer() - start_time
            fps = car_distance / tot_time

            # Output some stuff so we can watch.
            print("Max: %d at %d\tepsilon %f\t(%d)\t%f fps" %
                  (max_car_distance, t, epsilon, car_distance, fps))

            # Reset.
            car_distance = 0
            start_time = timeit.default_timer()

        # Save the model every 10,000 frames.
        if t % 5000 == 0:
            saver.save(session, "tensorData/model.ckpt")
            print("Saving model")


def process_minibatch(session, minibatch, predict, state):
    """This does the heavy lifting, aka, the training. It's super jacked."""
    X_train = []
    y_train = []
    # Loop through our batch and create arrays for X and y
    # so that we can fit our model at every step.
    for memory in minibatch:
        # Get stored values.
        old_state_m, action_m, reward_m, new_state_m = memory

        # Get prediction on old state.
        feed_dict = {
            state: old_state_m
        }
        old_qval = session.run([predict], feed_dict=feed_dict)

        # Get prediction on new state.
        feed_dict = {
            state: new_state_m
        }
        newQ = session.run([predict], feed_dict=feed_dict)

        # Update according to our best move
        maxQ = np.max(newQ)
        y = np.zeros((1, 3))
        y[:] = old_qval[:]
        # Check for terminal state.
        if reward_m != -500:  # non-terminal state
            update = (reward_m + (GAMMA * maxQ))
        else:  # terminal state
            update = reward_m
        # Update the value for the action we took.
        y[0][action_m] = update
        X_train.append(old_state_m.reshape(NUM_INPUT,))
        y_train.append(y.reshape(3,))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train


if __name__ == "__main__":
    nn_param = [180, 164]
    params = {
        "batchSize": 100,
        "buffer": 50000,
        "nn": nn_param
    }

    session, update, predict, state, input, labels = neural_net(NUM_INPUT, nn_param)

    train_net(session, update, predict, state, input, labels, params)
