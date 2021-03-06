from __future__ import print_function
import math
from time import sleep
import smbus
from nn import neural_net
from datetime import datetime
from rplidar import RPLidar
import numpy as np
import tensorflow as tf
import traceback
import socket
from io import StringIO

i2c_address = 0x37
i2c_bus = 1

VELOCITY = 20
NUM_SENSORS = 40

session, update, prediction, state, input, labels = neural_net(NUM_SENSORS, [180, 164])
saver = tf.train.Saver()
saver.restore(session, tf.train.latest_checkpoint("tensorData/"))

# Docstring
def send(sbus, out):
    for char in out:
        try:
            sbus.write_byte(i2c_address, ord(char))
        except:
            print('Loose Connection!')
            sleep(1)
            send(bus, out)

def transform(points):
    points = [pair(p) for p in points if (p[1] < 60 or p[1] > 300)] # This is a hack. Change it in v2.0!!!!
    readings = [600] * 120
    for point in points:
        readings[point[0]] = min(point[1], readings[point[0]])
    # Take minimum of the 3 points.
    readings_min = [600] * 40
    for i in range(40):
        readings_min = min(readings[3*i], readings[3*i+1], readings[3*i+2])
    # print("points:",points, readings)
    return np.array(readings_min)

def pair(point):
    return (int(math.floor(point[1] + 60) % 360), point[2] * 1.2)  # This is a hack. Fix it in the simulation.

def rplidar_init():
    
    try:
        lidar = RPLidar('/dev/ttyUSB1')
    except:
        lidar = RPLidar('/dev/ttyUSB0')
    
    lidar.start_motor()
    info = lidar.get_info()
    print(info)

    health = lidar.get_health()
    print(health)
    
    return lidar

#def dt_choice(dt_state):
#    i = np.argmin(dt_state, 0)
#    if i < 13 or i > 28:
#        return 0
#    elif i < 20:
#        return 20
#    return -20

def nn_choice(nn_state):
    nn_state = np.reshape(nn_state, (1, nn_state.shape[0]))
    feed_dict = {state: nn_state}
    action = np.argmax(session.run([prediction], feed_dict=feed_dict)) # Choose action.
    return -20 if action == 0 else (20 if action == 1 else 0)


bus = smbus.SMBus(i2c_bus)
while True:
    lidar = rplidar_init()

    sleep(1)

    try:
        for i, scan in enumerate(lidar.iter_scans()): # Read the LiDaR point cloud
            #sleep(1)
            nn_state = transform(scan) # Transform Point Cloud into suitable NP-Array
            action = nn_choice(nn_state) # Pass input through NeuralNet, then transform to degree
            print("Action: ", action)
	    
            output = str(action) + ',' + str(VELOCITY) + '.'
            print(output)
            send(bus, output)
            
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        lidar.stop_motor()
        lidar.disconnect()
