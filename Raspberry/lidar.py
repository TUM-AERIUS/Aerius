from rplidar import RPLidar
import time
from multiprocessing import Pipe
from threading import Thread, Event


class LidarController(object):
    """
    Asynchronous controller for the rplidar.
    """
    def __init__(self, port='/dev/ttyUSB0'):
        self.lidar = RPLidar(port)
        print('LidarController:', self.lidar.get_info())
        print('LidarController:', self.lidar.get_health())
        time.sleep(1)
        self._start_reading_data()

    def _create_background_reader(self, pipeout, stopsig):
        print('LidarController: background thread started.')
        for scan in self.lidar.iter_scans():
            if stopsig.is_set():
                print('LidarController: background thread finished.')
                return
            _, angles, dists = zip(*scan)
            pipeout.send((angles, dists))

    def _start_reading_data(self):
        self.pipein, pipeout = Pipe()
        self.stopsig = Event()
        Thread(target=self._create_background_reader, args=(pipeout, self.stopsig)).start()

    def stop(self):
        self.lidar.stop_motor()
        self.stopsig.set()

    def scan(self):
        """
        Read the latest rplidar scan. If now new scan if available, this function will block the calling thread until a
        scan is available.
        :param timeout If timeout is not specified then it will return immediately.
                If timeout is a number then this specifies the maximum time in seconds to block.
                If timeout is None then an infinite timeout is used
        :return: latest scan - (list of angles, list of distances)
        """
        scan = None, None
        is_first = True
        while self.pipein.poll() or is_first:
            scan = self.pipein.recv()
            is_first = False
        return scan

if __name__ == '__main__':
    lidar = LidarController()
    scan_num = 0
    for i in range(3):
        angles, dists = lidar.scan()
        time.sleep(1)
        print(angles, dists)
    lidar.stop()
