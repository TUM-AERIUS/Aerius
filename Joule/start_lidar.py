#!/usr/bin/python3
from rplidar import RPLidar

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


lidar = rplidar_init()

    # time.sleep(1000)


try:
    for i, scan in enumerate(lidar.iter_scans()): # Read the LiDaR point cloud
        #sleep(1)
        #nn_state = transform(scan) # Transform Point Cloud into suitable NP-Array
        #action = nn_choice(nn_state) # Pass input through NeuralNet, then transform to degree
        #print("Action: ", action)
        
        #output = str(action) + ',' + str(VELOCITY) + '.'
        print("Scan: ", i, "Data: ", scan)
        #send(bus, output)
except:
    print("error")

