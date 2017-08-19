from rplidar import RPLidar

try:
    RPLidar('/dev/ttyUSB1').stop_motor()
except:
    RPLidar('/dev/ttyUSB0').stop_motor()
