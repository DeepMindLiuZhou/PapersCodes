# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse, sys, os, math, re
import bpy
from mathutils import Vector, Matrix
import numpy as np
import cv2
import signal
from contextlib import contextmanager
from loguru import logger
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple
import random
class TimeoutException(Exception): pass

logger.info('Rendering started.')

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument(
    '--seed', type=int, default=0,
    help='number of views to be rendered')
parser.add_argument(
    '--views', type=int, default=4,
    help='number of views to be rendered')
parser.add_argument(
    'obj', type=str,
    help='Path to the obj file to be rendered.')
parser.add_argument(
    '--output_folder', type=str, default='/tmp',
    help='The path the output will be dumped to.')
parser.add_argument(
    '--scale', type=float, default=1,
    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument(
    '--format', type=str, default='PNG',
    help='Format of files generated. Either PNG or OPEN_EXR')

parser.add_argument(
    '--resolution', type=int, default=512,
    help='Resolution of the images.')
parser.add_argument(
    '--engine', type=str, default='CYCLES',
    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
parser.add_argument(
    '--gpu', type=int, default=0,
    help='gpu.')
parser.add_argument(
    '--animation_idx', type=int, default=0,
    help='The index of animation')

parser.add_argument(
    '--camera_option', type=str, default='fixed',
    help='Camera Options')
parser.add_argument(
    '--fixed_animation_length', type=int, default=-1,
    help='Set animation length to fixed number of framnes')
parser.add_argument(
    '--step_angle', type=int, default=3,
    help='Angle in degree for each step camera rotation')
parser.add_argument(
    '--downsample', type=int, default=1,
    help='Downsample ratio. No downsample by default')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


model_identifier = os.path.split(args.obj)[1].split('.')[0]
synset_idx = args.obj.split('/')[-2]

save_root = os.path.join(os.path.abspath(args.output_folder),  synset_idx, model_identifier, f'{args.animation_idx:03d}')

# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = args.engine# 'BLENDER_EEVEE'
render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
render.image_settings.file_format = args.format  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = args.resolution
render.resolution_y = args.resolution
render.resolution_percentage = 100
bpy.context.scene.cycles.filter_width = 0.01
bpy.context.scene.render.film_transparent = True
render_depth_normal = False
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.diffuse_bounces = 1
bpy.context.scene.cycles.glossy_bounces = 1
bpy.context.scene.cycles.transparent_max_bounces = 1
bpy.context.scene.cycles.transmission_bounces = 1
bpy.context.scene.cycles.samples = 16
bpy.context.scene.cycles.use_denoising = True
bpy.context.scene.cycles.denoiser = 'OPTIX'
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'


def enable_cuda_devices():
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.get_devices()
    # Attempt to set GPU device types if available
    for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
        try:
            cprefs.compute_device_type = compute_device_type
            print("Compute device selected: {0}".format(compute_device_type))
            break
        except TypeError:
            pass

    # Any CUDA/OPENCL devices?
    acceleratedTypes = ['CUDA', 'OPENCL', 'OPTIX']
    acceleratedTypes = ['CUDA', 'OPENCL']
    accelerated = any(device.type in acceleratedTypes for device in cprefs.devices)
    print('Accelerated render = {0}'.format(accelerated))

    # If we have CUDA/OPENCL devices, enable only them, otherwise enable
    # all devices (assumed to be CPU)
    print(cprefs.devices)
    for idx, device in enumerate(cprefs.devices):
        device.use = (not accelerated or device.type in acceleratedTypes)# and idx == args.gpu
        print('Device enabled ({type}) = {enabled}'.format(type=device.type, enabled=device.use))
    return accelerated

enable_cuda_devices()
context.active_object.select_set(True)
bpy.ops.object.delete()

# Import textured mesh
bpy.ops.object.select_all(action='DESELECT')

try:
    with time_limit(1000):
        imported_object = bpy.ops.import_scene.gltf(filepath=args.obj, merge_vertices=True, guess_original_bind_pose=False, bone_heuristic="TEMPERANCE")
except TimeoutException as e:
    print("Timed out finished!")
    exit()


# count animated frames
animation_names = []
ending_frame_list = {}
for k in bpy.data.actions.keys():
    matched_obj_name = ''
    for obj in bpy.context.selected_objects:
        if '_'+obj.name in k and len(obj.name) > len(matched_obj_name):
            matched_obj_name = obj.name
    a_name = k.replace('_'+matched_obj_name, '')
    a = bpy.data.actions[k]
    frame_start, frame_end = map(int, a.frame_range)
    logger.info(f'{k} | frame start: {frame_start}, frame end: {frame_end} | fps: {bpy.context.scene.render.fps}')
    if a_name not in animation_names:
        animation_names.append(a_name)
        ending_frame_list[a_name] = frame_end
    else:
        ending_frame_list[a_name] = max(frame_end, ending_frame_list[a_name])
        

        
selected_a_name = animation_names[args.animation_idx]
max_frame = ending_frame_list[selected_a_name]
for obj in bpy.context.selected_objects:
    if obj.animation_data is not None:
        obj_a_name = selected_a_name+'_'+obj.name
        if obj_a_name in bpy.data.actions:
            print('Found ', obj_a_name)
            obj.animation_data.action = bpy.data.actions[obj_a_name]
        else:
            print('Miss ', obj_a_name)
            
num_frames = args.fixed_animation_length if args.fixed_animation_length != -1 else max_frame
num_frames = num_frames // args.downsample

if num_frames == 0:
    print("No animation!")
    exit()

# from https://github.com/allenai/objaverse-xl/blob/main/scripts/rendering/blender_script.py
def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object):
    """Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT
def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object


def randomize_lighting() -> Dict[str, bpy.types.Object]:
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),
#         energy=random.choice([3, 4, 5]),
        energy=4,
    )

    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),
#         energy=random.choice([2, 3, 4]),
        energy=3,
    )

    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),
#         energy=random.choice([3, 4, 5]),
        energy=4,
    )

    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),
#         energy=random.choice([1, 2, 3]),
        energy=2,
    )

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )

def scene_bbox(
    single_obj = None, ignore_matrix = False
):
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for i in range(num_frames):
        bpy.context.scene.frame_set(i * args.downsample)
        for obj in get_scene_meshes() if single_obj is None else [single_obj]:
            found = True
            for coord in obj.bound_box:
                coord = Vector(coord)
                if not ignore_matrix:
                    coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
                
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)

def get_scene_meshes():
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def get_scene_root_objects():
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj
            
def normalize_scene():
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        None
    """
    if len(list(get_scene_root_objects())) > 1:
        # create an empty object to be used as a parent for all root objects
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)

        # parent all root objects to the empty object
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    logger.info(f"Scale: {scale}")
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    # unparent the camera
    bpy.data.objects["Camera"].parent = None

normalize_scene()

randomize_lighting()

# Place camera
cam = scene.objects['Camera']
cam.location = (0, 1.5, 0)  # radius equals to 1
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

cam_empty = bpy.data.objects.new("Empty", None)
cam_empty.location = (0, 0, 0)
cam.parent = cam_empty

scene.collection.objects.link(cam_empty)
context.view_layer.objects.active = cam_empty
cam_constraint.target = cam_empty

stepsize = 360.0 / args.views
rotation_mode = 'XYZ'


np.random.seed(args.seed)

if args.camera_option == "fixed":
    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'
    
    elevation_angle = 0.
    rotation_angle = 0.

    for view_idx in range(args.views):
        img_folder = os.path.join(save_root, f'{view_idx:03d}', 'img')
        mask_folder = os.path.join(save_root, f'{view_idx:03d}', 'mask')
        camera_folder = os.path.join(save_root, f'{view_idx:03d}', 'camera')

        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)
        os.makedirs(camera_folder, exist_ok=True)
        
        np.save(os.path.join(camera_folder, 'rotation'), np.array([rotation_angle + view_idx * stepsize for _ in range(num_frames)]))
        np.save(os.path.join(camera_folder, 'elevation'), np.array([elevation_angle for _ in range(num_frames)]))

        cam_empty.rotation_euler[2] = math.radians(rotation_angle + view_idx * stepsize)
        cam_empty.rotation_euler[0] = math.radians(elevation_angle)
        
        # save camera RT matrix
        rt_matrix = get_3x4_RT_matrix_from_blender(cam)
        rt_matrix_path = os.path.join(camera_folder, "rt_matrix.npy")
        np.save(rt_matrix_path, rt_matrix)
        for i in range(0, num_frames):
            bpy.context.scene.frame_set(i * args.downsample)
            render_file_path = os.path.join(img_folder,'%03d.png' % (i))
            scene.render.filepath = render_file_path
            bpy.ops.render.render(write_still=True)

        for i in range(0, num_frames):
            img = cv2.imread(os.path.join(img_folder, '%03d.png' % (i)), cv2.IMREAD_UNCHANGED)
            mask =  img[:, :, 3:4] / 255.0
            white_img = img[:, :, :3] * mask + np.ones_like(img[:, :, :3]) * (1 - mask) * 255
            white_img = np.clip(white_img, 0, 255)
            cv2.imwrite(os.path.join(img_folder, '%03d.jpg' % (i)), white_img)
            cv2.imwrite(os.path.join(mask_folder, '%03d.png'%(i)), img[:, :, 3])
            os.system('rm %s'%(os.path.join(img_folder, '%03d.png' % (i))))
        
elif args.camera_option == "random":
    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'
    
    for view_idx in range(args.views):
        elevation_angle = np.random.rand(1) * 35 - 5 # [-5, 30]
        rotation_angle = np.random.rand(1) * 360
        
        img_folder = os.path.join(save_root, f'{view_idx:03d}', 'img')
        mask_folder = os.path.join(save_root, f'{view_idx:03d}', 'mask')
        camera_folder = os.path.join(save_root, f'{view_idx:03d}', 'camera')

        os.makedirs(img_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)
        os.makedirs(camera_folder, exist_ok=True)
        
        np.save(os.path.join(camera_folder, 'rotation'), np.array([rotation_angle for _ in range(num_frames)]))
        np.save(os.path.join(camera_folder, 'elevation'), np.array([elevation_angle for _ in range(num_frames)]))
        
        cam_empty.rotation_euler[2] = math.radians(rotation_angle)
        cam_empty.rotation_euler[0] = math.radians(elevation_angle)
        
        # save camera RT matrix
        rt_matrix = get_3x4_RT_matrix_from_blender(cam)
        rt_matrix_path = os.path.join(camera_folder, "rt_matrix.npy")
        np.save(rt_matrix_path, rt_matrix)
        
        for i in range(0, num_frames):
            bpy.context.scene.frame_set(i * args.downsample)
            render_file_path = os.path.join(img_folder,'%03d.png' % (i))
            scene.render.filepath = render_file_path
            bpy.ops.render.render(write_still=True)

        for i in range(0, num_frames):
            img = cv2.imread(os.path.join(img_folder, '%03d.png' % (i)), cv2.IMREAD_UNCHANGED)
            mask =  img[:, :, 3:4] / 255.0
            white_img = img[:, :, :3] * mask + np.ones_like(img[:, :, :3]) * (1 - mask) * 255
            white_img = np.clip(white_img, 0, 255)
            cv2.imwrite(os.path.join(img_folder, '%03d.jpg' % (i)), white_img)
            cv2.imwrite(os.path.join(mask_folder, '%03d.png'%(i)), img[:, :, 3])
            os.system('rm %s'%(os.path.join(img_folder, '%03d.png' % (i))))
    
else:
    raise NotImplemented
