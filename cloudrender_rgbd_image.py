from cloudrender.libegl import EGLContext
import logging
import numpy as np
import sys
import os
import json
from cloudrender.render import SimplePointcloud, DirectionalLight
from cloudrender.camera import PerspectiveCameraModel
from cloudrender.camera.trajectory import Trajectory
from cloudrender.scene import Scene
from cloudrender.capturing import AsyncPBOCapture, DirectCapture
from videoio import VideoWriter
from OpenGL import GL as gl
from tqdm import tqdm
from PIL import Image
import trimesh
import argparse

# Initialize logging
def initialize_logging():
	logger = logging.getLogger("main_script")
	logger.setLevel(logging.INFO)
	return logger

# Initialize OpenGL context using EGLContext
def initialize_context(resolution):
	context = EGLContext()
	if not context.initialize(*resolution):
		print("Error during context initialization")
		sys.exit(0)
	return context

# Create and set up OpenGL framebuffers and renderbuffers
def setup_framebuffers(resolution):
	_main_cb, _main_db = gl.glGenRenderbuffers(2)
	viewport_width, viewport_height = resolution
	gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, _main_cb)
	gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA, viewport_width, viewport_height)
	gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, _main_db)
	gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, viewport_width, viewport_height)
	
	_main_fb = gl.glGenFramebuffers(1)
	gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, _main_fb)
	gl.glFramebufferRenderbuffer(gl.GL_DRAW_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_RENDERBUFFER, _main_cb)
	gl.glFramebufferRenderbuffer(gl.GL_DRAW_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, _main_db)
	gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, _main_fb)
	gl.glDrawBuffers([gl.GL_COLOR_ATTACHMENT0])
	return _main_fb

# Configure OpenGL settings
def setup_opengl(resolution):
	gl.glEnable(gl.GL_BLEND)
	gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
	gl.glClearColor(1.0, 1.0, 1.0, 0)
	gl.glViewport(0, 0, *resolution)
	gl.glEnable(gl.GL_DEPTH_TEST)
	gl.glDepthMask(gl.GL_TRUE)
	gl.glDepthFunc(gl.GL_LESS)
	gl.glDepthRange(0.0, 5.0)

# Create and set a position of the camera
def create_camera(resolution):
	camera = PerspectiveCameraModel()
	camera.init_intrinsics(resolution, fov=50.0, far=50, near=0.05)
	return camera

# Create a scene
def create_scene():
	return Scene()

# Load pointcloud and add to scene
def load_pointcloud(scene, camera, path_pointcloud):
	renderable_pc = SimplePointcloud(camera=camera)
	renderable_pc.generate_shadows = False
	renderable_pc.init_context()
	pointcloud = trimesh.load(path_pointcloud)
	renderable_pc.set_buffers(pointcloud)
	scene.add_object(renderable_pc)

# Add directional light with shadows to scene
def setup_lighting(scene):
	light = DirectionalLight(np.array([0.0, -1.0, -1.0]), np.array([0.8, 0.8, 0.8]))
	shadowmap_offset = -light.direction * 3
	shadowmap = scene.add_dirlight_with_shadow(
		light=light, shadowmap_texsize=(1024, 1024),
		shadowmap_worldsize=(4.0, 4.0, 10.0),
		shadowmap_center=np.array([0.0, 0.0, 0.0]) + shadowmap_offset
	)
	return shadowmap, shadowmap_offset

# Create camera trajectory
def create_camera_trajectory():
	camera_trajectory = Trajectory()
	camera_trajectory.set_trajectory(json.load( \
		open('{}/TRAJ_SUB4_MPI_Etage6_working_standing.json'.format(args.path_input_data)) \
	))
	camera_trajectory.refine_trajectory(time_step=1/30.0)
	return camera_trajectory

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

# Main drawing loop
def main_drawing_loop(resolution, fps, video_start_time, video_length_seconds):
	logger = initialize_logging()
	context = initialize_context(resolution)
	_main_fb = setup_framebuffers(resolution)
	setup_opengl(resolution)
	camera = create_camera(resolution)
	main_scene = create_scene()
	load_pointcloud(main_scene, camera, '{}/pointcloud.ply'.format(args.path_input_data))
	shadowmap, shadowmap_offset = setup_lighting(main_scene)
	camera_trajectory = create_camera_trajectory()

	if not os.path.exists('{}/rgb_frame'.format(args.path_input_data)):
		os.makedirs('{}/rgb_frame'.format(args.path_input_data))
	if not os.path.exists('{}/depth_frame'.format(args.path_input_data)):
		os.makedirs('{}/depth_frame'.format(args.path_input_data))

	import time
	cnt = 0
	with VideoWriter('{}/output_rgb.mp4'.format(args.path_input_data), resolution=resolution, fps=fps) as vw, \
			 DirectCapture(resolution) as capturing:
		for current_time in tqdm(np.arange(video_start_time, video_start_time + video_length_seconds, 1 / fps)):
			shadowmap.camera.init_extrinsics(pose=shadowmap_offset)
			camera_trajectory.apply(camera, current_time)
			gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
			main_scene.draw()
			gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, _main_fb)
			
			start_time = time.time()
			color = capturing.request_color()
			# print('comp_time (color): {:3f}ms'.format((time.time() - start_time) * 1000))
			if color is not None:
				image = Image.fromarray(color)
				image.save('{}/rgb_frame/{:06}.png'.format(args.path_input_data, cnt))
				vw.write(color)
				
			start_time = time.time()
			depth = capturing.request_depth()
			# print('comp_time (depth): {:3f}ms'.format((time.time() - start_time) * 1000))
			if depth is not None:
				depth_normalized = (depth * 1000).astype(np.uint16)
				image = Image.fromarray(depth_normalized)
				image.save('{}/depth_frame/{:06}.png'.format(args.path_input_data, cnt))
				
			cnt += 1
	
	logger.info("Done")

# Run the main drawing loop
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Render and save an RGBD image.')
	parser.add_argument('--path_input_data', type=str, default='path_input_data.ply', help='Path of input data.')
	parser.add_argument('--camera_fov', type=float, default=50.0, help='FoV of camera.')
	args = parser.parse_args()

	resolution = (1280, 720)
	fps = 5.0
	video_start_time = 0.0
	video_length_seconds = 1.0
	intrinsic_matrix  = compute_intrinsic_matrix(args.camera_fov, args.camera_fov, resolution[0], resolution[1])
	print('intrinsic_matrix: \n', intrinsic_matrix)
	main_drawing_loop(resolution, fps, video_start_time, video_length_seconds)
