import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
import pymunk.pygame_util

# PyGame init
width = 1000
height = 600
NUM_INPUTS = 40
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
space = pymunk.Space()
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.
show_sensors = True
draw_screen = True
random_press = False


def points(a, b):
    return [(a, -b), (a, b), (-a, b), (-a, -b)]


def car_is_crashed(readings):
    for i in range(NUM_INPUTS):
        if readings[i] == 1: return True
    return False


def sum_readings(readings):
    """Sum the number of non-zero readings."""
    tot = 0
    for i in readings:
        tot += i
    return tot


def make_sonar_arm(x, y, distance=50):
    spread = 12  # Default spread.
    distance_from_car = 10  # Gap before first sensor.
    arm_points = []
    # Make an arm. We build it flat because we'll rotate it about the
    # center later.
    for i in range(1, distance):
        arm_points.append((distance_from_car + x + (spread * i), y))

    return arm_points


def get_rotated_point(x_1, y_1, x_2, y_2, radians):
    # Rotate x_2, y_2 around x_1, y_1 by angle.
    cos = math.cos(radians)
    sin = math.sin(radians)

    x_change = (x_2 - x_1) * cos + (y_2 - y_1) * sin
    y_change = (y_1 - y_2) * cos - (x_1 - x_2) * sin

    new_x = x_change + x_1
    new_y = height - (y_change + y_1)

    return int(new_x), int(new_y)


def get_track_or_not(reading):
    colors = [THECOLORS['blue'], THECOLORS['orange'], THECOLORS['red'], THECOLORS['grey']]
    return 1 if reading in colors else 0


def get_arm_distance(arm, x, y, angle, offset):
    # Used to count the distance.
    i = 0

    # Look at each point and see if we've hit something.
    for point in arm:
        i += 1

        # Move the point to the right spot.
        rotated_p = get_rotated_point(x, y, point[0], point[1], angle + offset)

        # Check if we've hit something. Return the current i (distance)
        # if we did.

        if not (0 < rotated_p[0] < width and 0 < rotated_p[1] < height):
            return len(arm)  # Sensor is off the screen.
        else:
            obs = screen.get_at(rotated_p)
            if get_track_or_not(obs) != 0:
                return i

        if show_sensors:  # if straight
            color = (34, 122, 172) if len(arm) > 10 else (78, 127, 80)
            pygame.draw.circle(screen, color, rotated_p, 1)

    # Return the distance for the arm.
    return i


def get_sonar_readings(x, y, angle):
    readings = []
    """
    Instead of using a grid of boolean(ish) sensors, sonar readings
    simply return N "distance" readings, one for each sonar
    we're simulating. The distance is a count of the first non-zero
    reading starting at the object. For instance, if the fifth sensor
    in a sonar "arm" is non-zero, then that arm returns a distance of 5.
    """
    # Make our arms.
    cos10 = 10 * math.cos(angle)
    sin10 = 10 * math.sin(angle)
    arm = make_sonar_arm(x, y)
    # side_arm = make_sonar_arm(x - cos10, y - sin10, distance=10)

    # Rotate them and get readings.
    for i in range(NUM_INPUTS):
        offset = -1 + (2 / NUM_INPUTS) * i
        readings.append(get_arm_distance(arm, x, y, angle, offset))

    # Right and left ultrasonic sensors
    # min_left = math.inf
    # min_right = math.inf
    #
    # for i in range(NUM_INPUTS // 4):
    #     delta = -0.6 + (1.2 / (NUM_INPUTS // 4)) * i
    #     pi2 = (math.pi / 2)
    #
    #     left  = get_arm_distance(side_arm, x - cos10, y - sin10, angle + pi2, delta)
    #     right = get_arm_distance(side_arm, x - cos10, y - sin10, angle - pi2, delta)
    #
    #     if left  < min_left:  min_left  = left
    #     if right < min_right: min_right = right

    #readings.append(min_left)
    #readings.append(min_right)

    if show_sensors:
        pygame.display.update()

    return readings


class GameState:
    def __init__(self):
        # Global-ish.
        self.crashed = False

        self.given_x = 1
        self.given_y = 0
        self.given_angle = 0  # 0 till 2*pi

        # Physics stuff.
        self.space = space
        self.space.gravity = pymunk.Vec2d(0., 0.)

        # Create rewards
        self.old_reward = 0
        self.new_reward = 0

        self.create_car(100, 100, 0.5)  # Create the car.

        self.num_steps = 0  # Record steps.

        # Create walls.
        w30 = width + 30
        h30 = height + 30

        static = [
            pymunk.Segment(self.space.static_body, (-30, -30), (-30, h30), 1),
            pymunk.Segment(self.space.static_body, (-30, h30), (w30, h30), 1),
            pymunk.Segment(self.space.static_body, (w30, h30), (w30, -30), 1),
            pymunk.Segment(self.space.static_body, (-30, -30), (w30, -30), 1)
        ]

        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(static)

        # Create some obstacles
        self.obstacles = []
        self.generate_obstacles()

    def generate_obstacles(self):
        for obj in self.obstacles:
            self.space.remove(obj[0])
            self.space.remove(obj[1])

        self.obstacles = []
        obstacle_count = random.randint(5, 10)

        for i in range(obstacle_count):
            x = random.randint(150, width - 150)
            y = random.randint(150, height - 150)
            r = random.randint(20, 70)
            self.obstacles.append(self.create_obstacle(x, y, r))

        space.debug_draw(draw_options)

    def create_obstacle(self, x, y, r):
        c_body = pymunk.Body(body_type=pymunk.Body.DYNAMIC, mass=100, moment=10000)

        a = random.randint(50, 200)
        b = random.randint(50, 150)

        c_shape = pymunk.Circle(c_body, r) if random.randint(0, 2) == 0 \
            else  pymunk.Poly(c_body, points(a / 2, b / 2))

        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = THECOLORS["grey"]

        self.space.add(c_body, c_shape)

        return c_body, c_shape

    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_box(100000, [40, 20])

        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y

        self.car_shape = pymunk.Poly(self.car_body, points(20, 10))

        self.car_shape.color = THECOLORS["white"]
        self.car_shape.elasticity = 1.0

        self.car_body.angle = r

        self.car_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.space.add(self.car_body, self.car_shape)

    def frame_step(self, action):
        if   action == 0:  self.car_body.angle -= .15  # Turn Left
        elif action == 1:  self.car_body.angle += .15  # Turn Right

        # Move obstacles.
        if self.num_steps % 100 == 0:
            self.move_obstacles()

        old_x, old_y = self.car_body.position

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = 100 * driving_direction

        # Update the screen and stuff.
        screen.fill(THECOLORS["black"])
        space.debug_draw(draw_options)
        self.space.step(1. / 10)

        clock.tick(55)

        # Get the current location and the readings there.
        x, y = self.car_body.position
        readings = get_sonar_readings(x, y, self.car_body.angle)
        # add direction angle
        # cos_alfa = math.cos(self.car_body.angle - self.given_angle)
        # readings.append(cos_alfa)

        # car direction
        # d = ((x - old_x) ** 2 + (y - old_y) ** 2) ** .5
        # readings.append(self.car_body.angle % (2 * math.pi))
        # readings.append((x - old_x) / d)
        # readings.append((y - old_y) / d)

        # given direction
        # delta = (self.given_x ** 2 + self.given_y ** 2) ** .5

        # readings.append(self.given_angle % 2 * math.pi)
        # readings.append(self.given_x / delta)
        # readings.append(self.given_y / delta)

        # car position
        # readings.append(self.car_body.position.x)
        # readings.append(self.car_body.position.y)
        state = np.array([readings])
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:   self.given_angle += math.pi / 8
                if event.key == pygame.K_RIGHT:  self.given_angle -= math.pi / 8

                self.given_angle %= 2 * math.pi
                self.given_x = math.cos(self.given_angle)
                self.given_y = -math.sin(self.given_angle)

        if random_press and self.num_steps % 300 == 0:
            n = random.randint(0, 2)

            if n == 0: self.given_angle += math.pi / 8
            if n == 1: self.given_angle -= math.pi / 8

            self.given_angle %= 2 * math.pi
            self.given_x = + math.cos(self.given_angle)
            self.given_y = - math.sin(self.given_angle)

        # Set the reward.
        # Car crashed when any reading == 1
        if car_is_crashed(readings):
            self.crashed = True
            reward = -500
            self.recover_from_crash(driving_direction)
        # elif cos_alfa > 0.9:
        #     reward =
        else:
            # Higher readings are better, so return the sum.
            if random_press:
                self.old_reward = self.new_reward
                self.new_reward = 10 * (cos_alfa ** 3)
                reward = self.new_reward - self.old_reward
                if reward > 0 and cos_alfa > 0.95:
                    reward += 5 * cos_alfa
            else:
                reward = 0

        self.num_steps += 1

        if x > width - 10 or x < 10 or y > height - 10 or y < 10:
            self.generate_obstacles()

        if x > width - 10:
            self.car_body.position = 30, y

        if x < 10:
            self.car_body.position = width - 30, y

        if y > height - 10:
            self.car_body.position = x, 30

        if y < 10:
            self.car_body.position = x, height - 30

        white = 255, 255, 255
        radius = 40
        point1 = width - radius - 10, height - radius - 10
        point2 = point1[0] + self.given_x * radius, point1[1] + self.given_y * radius
        pygame.draw.circle(screen, white, point1, radius, 1)
        pygame.draw.line(screen, white, point1, point2)
        if draw_screen:
            pygame.display.flip()

        return reward, state

    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = random.randint(0, 10)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle[0].velocity = speed * direction

    def recover_from_crash(self, driving_direction):
        """
        We hit something, so recover.
        """
        while self.crashed:
            # Go backwards.
            self.car_body.velocity = -100 * driving_direction
            self.crashed = False
            for i in range(10):
                self.car_body.angle += .2  # Turn a little.
                screen.fill(THECOLORS["black"])
                space.debug_draw(draw_options)
                self.space.step(1. / 10)
                if draw_screen:
                    pygame.display.flip()
                clock.tick(55)


if __name__ == "__main__":
    game_state = GameState()
    while True:
        game_state.frame_step((random.randint(0, 2)))
