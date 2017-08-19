#!/usr/bin/python3
from rplidar import RPLidar

try:
    RPLidar('/dev/ttyUSB1').start_motor()
except:
    RPLidar('/dev/ttyUSB0').start_motor()
