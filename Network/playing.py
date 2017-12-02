"""
Once a model is learned, use this to play it.
"""

import carmunk
import numpy as np
from nn import neural_net
import tensorflow as tf

NUM_SENSORS = 40


def play(session, state, prediction):
    saver = tf.train.Saver()
    saver.restore(session, tf.train.latest_checkpoint("tensorData/"))

    car_distance = 0
    game_state = carmunk.GameState()

    # Do nothing to get initial.
    _, gamestate = game_state.frame_step((2))

    # Move.
    while True:
        car_distance += 1
        obstacle = False
        for i in range(NUM_SENSORS):
            if 15 < i < 30:
                if gamestate[0][i] < 10:
                    obstacle = True
                    break
            else:
                if gamestate[0][i] < 4:
                    obstacle = True
                    break

        # side sensors
        # if gamestate[0][40] <= 2 or gamestate[0][41] <= 2:
        #     obstacle = True

        if obstacle:
            feed_dict = {state: gamestate}
            action = np.argmax(session.run([prediction], feed_dict=feed_dict)) # Choose action.

        else:
            # car velocity vector
            # dx = gamestate[0][44]
            # dy = gamestate[0][45]
            #
            # given velocity vector
            # fx = gamestate[0][47]
            # fy = -gamestate[0][48]
            #
            # cos_angle = dx * fx + dy * fy
            # sin_angle = dx * fy - fx * dy
            #
            # if cos_angle < 0.998:
            #     action = 1 if sin_angle > 0 else 0
            # else: action = 2
            action = 0

        # Take action.
        _, gamestate = game_state.frame_step(action)

        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)


if __name__ == "__main__":
    session, update, prediction, state, input, labels = neural_net(NUM_SENSORS, [180, 164])

    play(session, state, prediction)
