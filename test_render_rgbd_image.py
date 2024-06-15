# On some systems, EGL does not start properly if OpenGL was already initialized, that's why it's better
# to keep EGLContext import on top
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

logger = logging.getLogger("main_script")
logger.setLevel(logging.INFO)

'''
##### SETUP logging configuration to INFO level
##### SET target resolution, framerate, video start time, and video length
##### INITIALIZE OpenGL context using EGLContext
'''

# This example shows how to:
# - render pointcloud
# - smoothly move the camera
# - dump rendered frames to a video

# First, let's set the target resolution, framerate, video length and initialize OpenGL context.
# We will use EGL offscreen rendering for that, but you can change it to whatever context you prefer (e.g. OsMesa, X-Server)
resolution = (1280, 720)
fps = 1.0
video_start_time = 6.0
video_length_seconds = 12.0
logger.info("Initializing EGL and OpenGL")
context = EGLContext()
if not context.initialize(*resolution):
    print("Error during context initialization")
    sys.exit(0)

'''
##### CREATE and set up OpenGL framebuffers and renderbuffers
##### CONFIGURE OpenGL settings
'''

# Now, let's create and set up OpenGL frame and renderbuffers
################### set color and depth buffer
_main_cb, _main_db = gl.glGenRenderbuffers(2)
viewport_width, viewport_height = resolution
gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, _main_cb)
gl.glRenderbufferStorage(
    gl.GL_RENDERBUFFER, gl.GL_RGBA,
    viewport_width, viewport_height
)
gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, _main_db)
gl.glRenderbufferStorage(
    gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24,
    viewport_width, viewport_height
)

################### set frame buffer
_main_fb = gl.glGenFramebuffers(1)
gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, _main_fb)
gl.glFramebufferRenderbuffer(
    gl.GL_DRAW_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
    gl.GL_RENDERBUFFER, _main_cb
)
gl.glFramebufferRenderbuffer(
    gl.GL_DRAW_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
    gl.GL_RENDERBUFFER, _main_db
)

################### set draw buffer
gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, _main_fb)
gl.glDrawBuffers([gl.GL_COLOR_ATTACHMENT0])

# Let's configure OpenGL
gl.glEnable(gl.GL_BLEND)
gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
gl.glClearColor(1.0, 1.0, 1.0, 0)
gl.glViewport(0, 0, *resolution)
gl.glEnable(gl.GL_DEPTH_TEST)
gl.glDepthMask(gl.GL_TRUE)
gl.glDepthFunc(gl.GL_LESS)
gl.glDepthRange(0.0, 5.0)

'''
##### CREATE and set up camera
##### CREATE a Scene object
'''

# Create and set a position of the camera
camera = PerspectiveCameraModel()
camera.init_intrinsics(resolution, fov=75, far=50)
camera.init_extrinsics(np.array([1, np.pi / 5, 0, 0]), np.array([0, -1, 2]))

# Create a scene
main_scene = Scene()

'''
##### LOAD pointcloud
##### ADD directional light with shadows to scene
##### CREATE camera trajectory
'''

# Load pointcloud
logger.info("Loading pointcloud")
renderable_pc = SimplePointcloud(camera=camera)
# Turn off shadow generation from pointcloud
renderable_pc.generate_shadows = False
renderable_pc.init_context()
pointcloud = trimesh_load_from_zip("test_assets/MPI_Etage6.zip", "*/pointcloud.ply")
renderable_pc.set_buffers(pointcloud)
main_scene.add_object(renderable_pc)

# Add a directional light with shadows for this scene
light = DirectionalLight(np.array([0.0, -1.0, -1.0]), np.array([0.8, 0.8, 0.8]))

# Create a 4x4x10 meter shadowmap with 1024x1024 texture buffer and center it above the model along the direction
# of the light. We will move the shadowmap with the model in the main loop
shadowmap_offset = -light.direction * 3
shadowmap = main_scene.add_dirlight_with_shadow(light=light, shadowmap_texsize=(1024, 1024),
                                                shadowmap_worldsize=(4.0, 4.0, 10.0),
                                                shadowmap_center=np.array([0.0, 0.0, 0.0]) + shadowmap_offset)

# Set camera trajectory and fill in spaces between keypoints with interpolation
logger.info("Creating camera trajectory")
camera_trajectory = Trajectory()
camera_trajectory.set_trajectory(json.load(open("test_assets/TRAJ_SUB4_MPI_Etage6_working_standing.json")))
camera_trajectory.refine_trajectory(time_step=1 / 30.0)
print(camera_trajectory)

'''
##### RUN main drawing loop
'''

### Main drawing loop ###
logger.info("Running the main drawing loop")

from PIL import Image
cnt = 0
with DirectCapture(resolution) as capturing:
    for current_time in tqdm(np.arange(video_start_time, video_start_time + video_length_seconds, 1 / fps)):
        # Update dynamic objects
        shadowmap.camera.init_extrinsics(pose=shadowmap_offset)
        # Move the camera along the trajectory
        camera_trajectory.apply(camera, current_time)

        # Clear OpenGL buffers
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # Draw the scene
        main_scene.draw()
        # Request color readout; optionally receive previous request
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, _main_fb)

        # gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        color = capturing.request_color()
        if color is not None:
            image = Image.fromarray(color)
            image.save('/Titan/dataset/cloudrender/test_assets/rgb_frame/{:06}.png'.format(cnt))
    
        # gl.glReadBuffer(gl.GL_DEPTH_ATTACHMENT)
        depth = capturing.request_depth()
        if depth is not None:
            depth_normalized = (depth * 1000).astype(np.uint16)
            image = Image.fromarray(depth_normalized)
            image.save('/Titan/dataset/cloudrender/test_assets/depth_frame/{:06}.png'.format(cnt))

        cnt += 1

logger.info("Done")
