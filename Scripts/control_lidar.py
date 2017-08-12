from rplidar import RPLidar
import sys

connection = '/dev/ttyUSB0'
connection = '/dev/tty.SLAB_USBtoUART'

if len(sys.argv) > 1:
    cmd = sys.argv[1]
    if cmd == 'start':
        RPLidar(connection).start_motor()
    elif cmd == 'stop':
        RPLidar(connection).stop_motor()
    else:
        raise ValueError("Unknown command ...")



