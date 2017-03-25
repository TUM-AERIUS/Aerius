"""
Once a model is learned, use this to play it.
"""

import carmunk
import numpy as np
from nn import neural_net
import tensorflow as tf

NUM_SENSORS = 47


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
        obstacle = False
        for i in range(40):
            if gameState[0][i] < 10:
                obstacle = True
                break

        if obstacle:
            feed_dict = {
                state: gameState
            }

            # Choose action.
            action = np.argmax(session.run([prediction], feed_dict=feed_dict))

        else:
            # car velocity vector
            dx = gameState[0][42]
            dy = gameState[0][43]
            # given velocity vector
            fx = gameState[0][45]
            fy = -gameState[0][46]

            cos_angle = dx*fx + dy*fy
            sin_angle = dx*fy - fx*dy

            if cos_angle < 0.999:
                if sin_angle > 0:
                    action = 1
                else:
                    action = 0
            else:
                action = 2

        # Take action.
        _, gameState = game_state.frame_step(action)

        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)


if __name__ == "__main__":
    session, update, prediction, state, input, labels = neural_net(NUM_SENSORS, [180, 164])

    play(session, state, prediction)
