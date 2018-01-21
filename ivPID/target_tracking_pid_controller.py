import PID
import math

"""
Usage:

	controller = TargetTrackingPID(desired_x=100, desired_y=100,steering_angle_d=5)
	
	while driving:
		steering_angle += self.update(target_x, target_y)

"""



class TargetTrackingPID:
	"""
	Computes the controls for keeping the target position in the desired position on the image.
	It makes use only of the desired_x and target_x to compute the difference of image 
	"""


	def __init__(
		self, 
		desired_x, 
		desired_y, 
		steering_angle_d = 5):
		self.desired_x      = desired_x
		self.desired_y      = desired_y
		self.steering_angle_d = steering_angle_d # degrees

		self.PID = PID.PID(P = 0.2, I=1, D = 0.001)
		self.PID.clear()
		self.PID.setSampleTime(0.1)	# seconds
		self.PID.SetPoint = 0 		# target_x-desired_x should be 0, object is in center view then.



	def get_action(self, target_x, target_y):
		"""
			Given the target coordinates in the image, computes the actions in order to keep the target
		   	as close as possible to the desired position.

			Args:
				target_x     (int): x position of the target in the image
				target_y     (int): y position of the target in the image
			Returns:
				int: The steering angle: Positive value means steering to the right.
		"""

		# error is difference between the target's and desired's x positions

		dist = self.desired_x-self.target_x

		self.PID.update(dist)

		if self.output > 0:
			return self.steering_angle_d
		else:
			return -self.steering_angle_d
