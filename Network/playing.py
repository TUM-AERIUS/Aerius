"""
Once a model is learned, use this to play it.
"""

import carmunk
import numpy as np
from nn import neural_net
import tensorflow as tf

NUM_SENSORS = 43


def play(session, state, prediction):

    saver = tf.train.Saver()
    saver.restore(session, tf.train.latest_checkpoint("tensorData/"))

    car_distance = 0
    game_state = carmunk.GameState()

    # Do nothing to get initial.
    _, gameState = game_state.frame_step((2))

    # Move.
    while True:
        car_distance += 1

        feed_dict = {
            state: gameState
        }

        # Choose action.
        action = np.argmax(session.run([prediction], feed_dict=feed_dict))

        # Take action.
        _, gameState = game_state.frame_step(action)

        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)


if __name__ == "__main__":
    session, update, prediction, state, input, labels = neural_net(NUM_SENSORS, [180, 164])

    play(session, state, prediction)
