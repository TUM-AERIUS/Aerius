import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Joule import lidar_client


class LidarVisualizer:
    MIN_ANGLE = 50.0
    MAX_ANGLE = 130.0
    POINTS_PER_BEAM = 50
    N_LIDAR = 40
    MAX_RANGE = 600.0

    def __init__(self, conn):
        self.conn = conn

    def start(self):
        fig = plt.figure()
        ax = plt.gca()

        self.line_ani = animation.FuncAnimation(
            fig, self._on_draw, 10000, fargs=(ax,), interval=1, blit=True)
        plt.show()

    def _on_draw(self, index, ax):
        sensor_angles = np.linspace(self.MIN_ANGLE / 180.0 * np.pi,
                                    self.MAX_ANGLE / 180.0 * np.pi, num=self.N_LIDAR)
        sensor_ranges = lidar_client.receive_data(self.conn)
        sensor_ranges = np.array(sensor_ranges, dtype="float32")


        plt.cla()
        plt.xlim(-self.MAX_RANGE, self.MAX_RANGE)
        plt.ylim(0.0, self.MAX_RANGE)

        origin = np.array([0.0, 0.0], dtype="float32")
        sensor_directions = np.stack([np.cos(sensor_angles), np.sin(sensor_angles)], axis=-1)
        sensor_end_points = sensor_directions * sensor_ranges[:, np.newaxis]
        sensor_end_points = np.array(sensor_end_points, dtype="float32")

        # TODO: enable this if you want multiple points
        # sensor_points = []
        # for point in sensor_end_points:
            # ray_x = np.linspace(0.0, point[0], num=self.POINTS_PER_BEAM)
            # ray_y = np.linspace(0.0, point[1], num=self.POINTS_PER_BEAM)
            # ray_points = np.stack([ray_x, ray_y], axis=1)
            # sensor_points.append(ray_points)

        # sensor_points = np.concatenate(sensor_points, axis=0)

        ax.plot(origin[0], origin[1], 'ro')
        ax.scatter(sensor_end_points[:, 0], sensor_end_points[:, 1], s=1)

        return ax,


if __name__ == "__main__":
    conn = lidar_client.create_connection()
    visualizer = LidarVisualizer(conn)
    visualizer.start()
