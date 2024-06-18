#!/usr/bin/python3
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as ROS_Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2 as ROS_PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import argparse

import os
import random
import time
from enum import Enum

import numpy as np
from PIL import Image

from scipy.spatial.transform import Rotation

from cloudrender.capturing import DirectCapture
from OpenGL import GL as gl
import cloudrender_rgbd_image as cr_rgbd
from cloudrender.camera import PerspectiveCameraModel

default_sim_settings = {
	"frame_rate": 30, # image frame rate
	"width": 640, # horizontal resolution
	"height": 360, # vertical resolution
	"fov": 50.0, # horizontal FOV
	"far_plane": 15.0,
	"near_plane": 0.05,
	"camera_offset_z": 0, # camera z-offset
	"camera_info": True,
	"color_sensor": True,  # RGB sensor
	"depth_sensor": True,  # depth sensor
	"semantic_sensor": False,  # semantic sensor
	"scene": "/Titan/dataset/cloudrender/test_canteen/pointcloud.ply",
}

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default=default_sim_settings["scene"])
args = parser.parse_args()

def compute_intrinsic_matrix(h_fov_deg, v_fov_deg, width, height):
	h_fov = np.radians(h_fov_deg)
	v_fov = np.radians(v_fov_deg)
	fx = width / (2 * np.tan(h_fov / 2))
	fy = height / (2 * np.tan(v_fov / 2))
	cx = width / 2
	cy = height / 2
	intrinsic_matrix = np.array([
			[fx, 0,  cx],
			[0,  fy, cy],
			[0,  0,  1]
	])
	return intrinsic_matrix

def make_settings():
	settings = default_sim_settings.copy()
	settings["scene"] = args.scene
	return settings

class DemoRunnerType(Enum):
	BENCHMARK = 1
	EXAMPLE = 2
	AB_TEST = 3

class ABTestGroup(Enum):
	CONTROL = 1
	TEST = 2

class DemoRunner:
	def __init__(self, sim_settings, simulator_demo_type):
		self.set_sim_settings(sim_settings)

	def set_sim_settings(self, sim_settings):
		self._sim_settings = sim_settings.copy()

	def setup_opengl(self):
		self.gl_resolution = (self._sim_settings['width'], self._sim_settings['height'])
		self.gl_logger = cr_rgbd.initialize_logging()
		self.gl_context = cr_rgbd.initialize_context(self.gl_resolution)
		self.gl_main_fb = cr_rgbd.setup_framebuffers(self.gl_resolution)
		cr_rgbd.setup_opengl(self.gl_resolution)
		self.gl_camera = PerspectiveCameraModel()
		self.gl_camera.init_intrinsics(self.gl_resolution, fov=self._sim_settings['fov'], \
																   far=self._sim_settings['far_plane'], near=self._sim_settings['near_plane'])
		cr_rgbd.create_camera(self.gl_resolution)
		self.gl_main_scene = cr_rgbd.create_scene()
		cr_rgbd.load_pointcloud(self.gl_main_scene, self.gl_camera, self._sim_settings['scene'])
		self.gl_shadowmap, self.gl_shadowmap_offset = cr_rgbd.setup_lighting(self.gl_main_scene)
		print('Finish loading pointcloud')

	def publish_camera_info(self, msg):
		msg.header.stamp = rospy.Time.from_sec(self.time)		
		self.camera_info_pub.publish(msg)

	def publish_color_observation(self, color_obs):
		from cv_bridge import CvBridge
		bridge = CvBridge()
		self.color_image = bridge.cv2_to_imgmsg(color_obs, encoding='rgb8', header=self.color_image.header)
		self.color_image.header.stamp = rospy.Time.from_sec(self.time)
		self.color_image_pub.publish(self.color_image)

	def publish_depth_observation(self, depth_obs):
		from cv_bridge import CvBridge
		bridge = CvBridge()
		self.depth_image = bridge.cv2_to_imgmsg(depth_obs, encoding="32FC1", header=self.depth_image.header)
		self.depth_image.header.stamp = rospy.Time.from_sec(self.time)
		self.depth_image_pub.publish(self.depth_image)

	def publish_semantic_observation(self, semantic_obs):
		pass
		# semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
		# semantic_img.putpalette(d3_40_colors_rgb.flatten())
		# semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
		# self.semantic_image.data = np.array(semantic_img.convert("RGB")).tobytes()
		# self.semantic_image.header.stamp = rospy.Time.from_sec(self.time)
		# self.semantic_image_pub.publish(self.semantic_image)

	def state_estimation_callback(self, msg):
		self.time = msg.header.stamp.to_sec()
		orientation = msg.pose.pose.orientation
		self.quat_w2b = np.array([orientation.w, orientation.x, orientation.y, orientation.z])
		self.trans_w2b = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])

		R_w2b = Rotation.from_quat(np.roll(self.quat_w2b, -1)).as_matrix()
		R_offset = Rotation.from_quat(np.roll([0.5, 0.5, -0.5, -0.5], -1)).as_matrix()
		R_w2c = R_w2b @ R_offset
		self.quat_w2c = np.roll(Rotation.from_matrix(R_w2c).as_quat(), 1)
		self.trans_w2c = self.trans_w2b
		
	def listener(self):
		print('Start listening')
		rospy.init_node("cloudrender_online")

		self.quat_w2c = np.array([0.5, 0.5, -0.5, -0.5])
		self.trans_w2c = np.array([0.0, 0.0, 1.412282849397663])
		rospy.Subscriber("/state_estimation", Odometry, self.state_estimation_callback)	
		self.time = 0

		if self._sim_settings["camera_info"]:
			self.camera_info_pub = rospy.Publisher("/cloudrender_camera/camera_info", CameraInfo, queue_size=2)

		if self._sim_settings["color_sensor"]:
			self.color_image_pub = rospy.Publisher("/cloudrender_camera/color/image", ROS_Image, queue_size=2)
			self.color_image = ROS_Image()
			self.color_image.header.frame_id = "cloudrender_camera"
			self.color_image.height = self._sim_settings["height"]
			self.color_image.width  = self._sim_settings["width"]
			self.color_image.encoding = "rgb8"
			self.color_image.step = 3 * self.color_image.width
			self.color_image.is_bigendian = False

		if self._sim_settings["depth_sensor"]:
			self.depth_image_pub = rospy.Publisher("/cloudrender_camera/depth/image", ROS_Image, queue_size=2)
			self.depth_image = ROS_Image()
			self.depth_image.header.frame_id = "cloudrender_camera"
			self.depth_image.height = self._sim_settings["height"]
			self.depth_image.width  = self._sim_settings["width"]
			self.depth_image.encoding = "mono8"
			self.depth_image.step = self.color_image.width
			self.depth_image.is_bigendian = False

		if self._sim_settings["semantic_sensor"]:
			self.semantic_image_pub = rospy.Publisher("/cloudrender_camera/semantic/image", ROS_Image, queue_size=2)
			self.semantic_image = ROS_Image()
			self.semantic_image.header.frame_id = "cloudrender_camera"
			self.semantic_image.height = self._sim_settings["height"]
			self.semantic_image.width  = self._sim_settings["width"]
			self.semantic_image.encoding = "rgb8"
			self.semantic_image.step = 3 * self.color_image.width
			self.semantic_image.is_bigendian = False

		r = rospy.Rate(default_sim_settings["frame_rate"])
		while not rospy.is_shutdown():
			with DirectCapture(self.gl_resolution) as capturing:
				self.gl_shadowmap.camera.init_extrinsics(pose=self.gl_shadowmap_offset)
				self.gl_camera.init_extrinsics(self.quat_w2c, self.trans_w2c)
				gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
				self.gl_main_scene.draw()
				gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.gl_main_fb)

				if self._sim_settings["camera_info"]:
					K = compute_intrinsic_matrix(\
						self._sim_settings['fov'], self._sim_settings['fov'], \
						self.gl_resolution[0], self.gl_resolution[1])
					camera_info_msg = CameraInfo()
					camera_info_msg.header.frame_id = 'cloudrender_camera'
					camera_info_msg.height = self._sim_settings['height']
					camera_info_msg.width = self._sim_settings['width']
					camera_info_msg.distortion_model = 'plumb_bob'
					camera_info_msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
					camera_info_msg.K = [K[0][0], 0, K[0][2], 0, K[1][1], K[1][2], 0, 0, 1]					
					camera_info_msg.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]   
					camera_info_msg.P = [K[0][0], 0, K[0][2], 0, 0, K[1][1], K[1][2], 0, 0, 0, 1, 0]
					self.publish_camera_info(camera_info_msg)

				if self._sim_settings["color_sensor"]:
					color_img = capturing.request_color()
					if color_img is not None:
						self.publish_color_observation(color_img)

				if self._sim_settings["depth_sensor"]:
					depth_img = capturing.request_depth()
					if depth_img is not None:
						near_plane = self._sim_settings['near_plane']
						far_plane = self._sim_settings['far_plane']
						linear_depth_buffer = (2.0 * near_plane * far_plane) / \
																	(far_plane + near_plane - (2.0 * depth_img - 1.0) * (far_plane - near_plane))
						self.publish_depth_observation(linear_depth_buffer)

				if self._sim_settings["semantic_sensor"]:
					pass
					# self.publish_semantic_observation(semantic_img)

				print("Publishing at time: " + str(self.time))
				r.sleep()

if __name__ == "__main__":
	settings = make_settings()
	print('settings: \n', settings)

	demo_runner = DemoRunner(settings, DemoRunnerType.EXAMPLE)
	demo_runner.setup_opengl()
	demo_runner.listener()
