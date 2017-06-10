import math
# import smbus
from nn import neural_net
from time import sleep
from datetime import datetime
from rplidar import RPLidar

i2c_address = 0x37
i2c_bus = 6

VELOCITY = 7

# session, update, prediction, state, input, labels = neural_net(NUM_SENSORS, [180, 164])
# saver = tf.train.Saver()
# saver.restore(session, tf.train.latest_checkpoint("tensorData/"))

# def send(bus, out):
    # for c in out:
        # try: bus.write_byte(i2c_address, ord(c))
        # except:
            # print('Loose Connection!')
            # sleep(1)
            # send(bus, out)

# def transform(points):
#     points = [pair(p) for p in points if p < 20 or p > 340]
#     readings = [50] * 49
#     for p in points:
#         readings[p[0]] = min(p[1], readings[p[0]])
#     return np.array(readings)

def pair(p):
    return (math.floor(p[1] + 20) % 360, p[2] / 100)

def rplidar_init():
    lidar = RPLidar('/dev/ttyUSB0')

    info = lidar.get_info()
    print(info)

    health = lidar.get_health()
    print(health)

    return lidar

# def nn_choice(nn_state):
    # feed_dict = {state: nn_state}
    # action = np.argmax(session.run([prediction], feed_dict=feed_dict)) # Choose action.
    # return -20 if action == 0 else (20 if action == 1 else 0)


# bus = smbus.SMBus(i2c_bus)
lidar = rplidar_init()

time.sleep(1000)

for i, scan in enumerate(lidar.iter_scans()): # Read the LiDaR point cloud
    print('%d: Got %d measurments' % (i, len(scan)))

    # nn_state = transform(scans) # Transform Point Cloud into suitable NP-Array
    # action = nn_choice(nn_state) # Pass input through NeuralNet, then transform to degree
    # print(action)

    # output = str(action) + ',' + str(VELOCITY) + '.'
    # send(bus, output)

lidar.stop()
lidar.stop_motor()
lidar.disconnect()
