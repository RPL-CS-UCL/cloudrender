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
from cloudrender.utils import trimesh_load_from_zip
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Render and save an RGBD image.')
parser.add_argument('--path_input_data', type=str, default='path_input_data.ply', help='Path of input data.')
args = parser.parse_args()

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
	camera.init_intrinsics(resolution, fov=75, far=50)
	return camera

# Create a scene
def create_scene():
	return Scene()

# Load pointcloud and add to scene
def load_pointcloud(scene, camera):
	renderable_pc = SimplePointcloud(camera=camera)
	renderable_pc.generate_shadows = False
	renderable_pc.init_context()
	import trimesh
	pointcloud = trimesh.load("test_assets/MPI_Etage6/pointcloud.ply")
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
	camera_trajectory.set_trajectory(json.load(open("test_assets/TRAJ_SUB4_MPI_Etage6_working_standing.json")))
	camera_trajectory.refine_trajectory(time_step=1 / 30.0)
	return camera_trajectory

# Main drawing loop
def main_drawing_loop(resolution, fps, video_start_time, video_length_seconds):
	logger = initialize_logging()
	context = initialize_context(resolution)
	_main_fb = setup_framebuffers(resolution)
	setup_opengl(resolution)
	camera = create_camera(resolution)
	main_scene = create_scene()
	load_pointcloud(main_scene, camera)
	shadowmap, shadowmap_offset = setup_lighting(main_scene)
	camera_trajectory = create_camera_trajectory()
	
	cnt = 0
	with DirectCapture(resolution) as capturing:
		for current_time in tqdm(np.arange(video_start_time, video_start_time + video_length_seconds, 1 / fps)):
			shadowmap.camera.init_extrinsics(pose=shadowmap_offset)
			camera_trajectory.apply(camera, current_time)
			gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
			main_scene.draw()
			gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, _main_fb)
			
			color = capturing.request_color()
			if color is not None:
				image = Image.fromarray(color)
				image.save('/Titan/dataset/cloudrender/test_assets/rgb_frame/{:06}.png'.format(cnt))
				
			depth = capturing.request_depth()
			if depth is not None:
				depth_normalized = (depth * 1000).astype(np.uint16)
				image = Image.fromarray(depth_normalized)
				image.save('/Titan/dataset/cloudrender/test_assets/depth_frame/{:06}.png'.format(cnt))
				
			cnt += 1
	
	logger.info("Done")

# Run the main drawing loop
if __name__ == "__main__":
	resolution = (1280, 720)
	fps = 1.0
	video_start_time = 0.0
	video_length_seconds = 20.0
	main_drawing_loop(resolution, fps, video_start_time, video_length_seconds)
