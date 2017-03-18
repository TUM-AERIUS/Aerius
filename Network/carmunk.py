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
height = 700
NUM_INPUTS = 40
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
space = pymunk.Space()
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.
show_sensors = False
draw_screen = True
random_press = False


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

        # Create the car.
        self.create_car(100, 100, 0.5)

        # Record steps.
        self.num_steps = 0

        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (-30, -30), (-30, height+30), 1),
            pymunk.Segment(
                self.space.static_body,
                (-30, height + 30), (width + 30, height + 30), 1),
            pymunk.Segment(
                self.space.static_body,
                (width+30, height+30), (width+30, -30), 1),
            pymunk.Segment(
                self.space.static_body,
                (-30, -30), (width+30, -30), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(static)

        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []
        self.generate_obstacles()

        # Create a cat.
        self.create_cat()

    def generate_obstacles(self):
        for obj in self.obstacles:
            self.space.remove(obj[0])
            self.space.remove(obj[1])
        self.obstacles = []
        obstacle_count = random.randint(5, 10)
        for i in range(obstacle_count):
            x = random.randint(50, width - 50)
            y = random.randint(50, height - 50)
            r = random.randint(20, 70)
            self.obstacles.append(self.create_obstacle(x, y, r))
        space.debug_draw(draw_options)

    def create_obstacle(self, x, y, r):
        c_body = pymunk.Body(body_type=pymunk.Body.DYNAMIC, mass=100, moment=1)
        # TODO create squares
        c_shape = pymunk.Circle(c_body, r)
        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = THECOLORS["blue"]
        self.space.add(c_body, c_shape)
        return c_body, c_shape

    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50, height - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 30)
        self.cat_shape.color = THECOLORS["orange"]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        self.direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.space.add(self.cat_body, self.cat_shape)

    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, 25)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        self.driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.space.add(self.car_body, self.car_shape)

    def frame_step(self, action):
        if action == 0:  # Turn left.
            self.car_body.angle -= .1
        elif action == 1:  # Turn right.
            self.car_body.angle += .1

        # Move obstacles.
        if self.num_steps % 20 == 0:
            self.move_obstacles()

        # Move cat.
        if self.num_steps % 5 == 0:
            self.move_cat()

        old_x, old_y = self.car_body.position

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = 100 * driving_direction

        # Update the screen and stuff.
        screen.fill(THECOLORS["black"])
        space.debug_draw(draw_options)
        self.space.step(1./10)

        clock.tick(25)

        # Get the current location and the readings there.
        x, y = self.car_body.position
        readings = self.get_sonar_readings(x, y, self.car_body.angle)
        # add direction angle
        cos_alfa = math.cos(self.car_body.angle - self.given_angle)
        readings.append(cos_alfa)
        readings.append(self.car_body.angle % 2*math.pi)
        readings.append((x-old_x)/math.sqrt(math.pow(x-old_x, 2) + math.pow(y-old_y, 2)))
        readings.append((y-old_y)/math.sqrt(math.pow(x-old_x, 2) + math.pow(y-old_y, 2)))
        readings.append(self.given_angle % 2*math.pi)
        readings.append(self.given_x / math.sqrt(math.pow(self.given_x, 2) + math.pow(self.given_y, 2)))
        readings.append(self.given_y / math.sqrt(math.pow(self.given_x, 2) + math.pow(self.given_y, 2)))
        state = np.array([readings])

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.given_angle += math.pi / 8
                if event.key == pygame.K_RIGHT:
                    self.given_angle -= math.pi / 8
                self.given_angle %= 2*math.pi
                self.given_x = math.cos(self.given_angle)
                self.given_y = -math.sin(self.given_angle)

        if random_press and self.num_steps % 300 == 0:
            n = random.randint(0, 2)
            if n == 0:
                self.given_angle += math.pi / 8
            if n == 1:
                self.given_angle -= math.pi / 8
            self.given_angle %= 2 * math.pi
            self.given_x = math.cos(self.given_angle)
            self.given_y = -math.sin(self.given_angle)

        # Set the reward.
        # Car crashed when any reading == 1
        if self.car_is_crashed(readings):
            self.crashed = True
            reward = -500
            self.recover_from_crash(driving_direction)
        # elif cos_alfa > 0.9:
        #     reward =
        else:
            # Higher readings are better, so return the sum.
            self.old_reward = self.new_reward
            self.new_reward = 10 * math.pow(cos_alfa, 3)
            reward = self.new_reward - self.old_reward
            if reward > 0 and cos_alfa > 0.95:
                reward += 5 * cos_alfa

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
            speed = random.randint(0, 20)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle[0].velocity = speed * direction

    def move_cat(self):
        speed = random.randint(20, 200)
        self.cat_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.cat_body.velocity = speed * direction

    def car_is_crashed(self, readings):
        for i in range(40):
            if readings[i] == 1:
                return True
        return False

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
                self.space.step(1./10)
                if draw_screen:
                    pygame.display.flip()
                clock.tick(25)

    def sum_readings(self, readings):
        """Sum the number of non-zero readings."""
        tot = 0
        for i in readings:
            tot += i
        return tot

    def get_sonar_readings(self, x, y, angle):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make our arms.
        arm = self.make_sonar_arm(x, y)
        # arm_middle = arm_left
        # arm_right = arm_left

        # Rotate them and get readings.
        for i in range(NUM_INPUTS):
            offset = -1 + (2 / NUM_INPUTS) * i
            readings.append(self.get_arm_distance(arm, x, y, angle, offset))

        if show_sensors:
            pygame.display.update()

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return len(arm)  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        # Return the distance for the arm.
        return i

    def make_sonar_arm(self, x, y):
        spread = 12  # Default spread.
        distance = 15  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 50):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        if reading == THECOLORS['blue'] or reading == THECOLORS['orange'] or reading == THECOLORS['red']:
            return 1
        else:
            return 0

if __name__ == "__main__":
    game_state = GameState()
    while True:
        game_state.frame_step((random.randint(0, 2)))
