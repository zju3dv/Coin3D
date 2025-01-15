import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from pathlib import Path
import bmesh

from mathutils import Vector, Matrix
import numpy as np
from io import BytesIO
import bpy
from mathutils import Vector
import pickle
import json
from urllib import request, parse
import random
import base64

def load_object(object_path: str) -> None:
    """Loads a 3D model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")

def az_el_to_points(azimuths, elevations):
    x = np.cos(azimuths)*np.cos(elevations)
    y = np.sin(azimuths)*np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x,y,z],-1) #

def points_to_az_el_dist(location):
    dist = np.linalg.norm(location)
    location /= dist
    x, y, z = location
    ele = np.arcsin(z)
    azi = np.arctan2(y, x)
    return azi, ele, dist

def set_camera(camera, az, el, dist):
    # az, el in degree
    distances = dist

    azimuths = np.deg2rad(az).astype(np.float32)
    elevations = np.deg2rad(el).astype(np.float32)

    cam_pts = az_el_to_points(azimuths, elevations) * distances
    x, y, z = cam_pts
    camera.location = x, y, z


def get_camera(camera_name):
    context = bpy.context
    scene = context.scene
    # 获取或创建Empty对象
    empty = scene.objects.get("Empty")
    if empty is None:
        empty = bpy.data.objects.new("Empty", None)
        scene.collection.objects.link(empty)

    # 尝试获取相机
    camera = scene.objects.get(camera_name)
    if camera is None:
        # 创建新的相机
        camera_data = bpy.data.cameras.new(name=camera_name)
        camera = bpy.data.objects.new(camera_name, camera_data)
        context.collection.objects.link(camera)
        camera.location = (0, 1.2, 0)
    camera.data.lens = 35
    camera.data.sensor_width = 32

    constrs = camera.constraints.get('Track To', None)
    if constrs is None:
        cam_constraint = camera.constraints.new(type='TRACK_TO')
        cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        cam_constraint.up_axis = 'UP_Y'
        # 设置约束的目标
        cam_constraint.target = empty
    return camera


def init_global(context):
    scene = context.scene
    render = scene.render

    cam = get_camera("Camera")
    set_camera(camera=cam, az=0, el=30, dist=1.5)
    bpy.context.scene.camera = cam

    render.engine = "CYCLES"
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = 256
    render.resolution_y = 256
    render.resolution_percentage = 100

    scene.cycles.device = "GPU"
    scene.cycles.samples = 128
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True

    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    # Set the device_type
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "NONE" # or "OPENCL"
    bpy.context.scene.cycles.tile_size = 8192

    world_tree = bpy.context.scene.world.node_tree
    back_node = world_tree.nodes['Background']
    env_light = 0.5
    back_node.inputs['Color'].default_value = Vector([env_light, env_light, env_light, 1.0])
    back_node.inputs['Strength'].default_value = 1.0

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent and isinstance(obj.data, (bpy.types.Mesh, bpy.types.Light)):
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def selected_root_objects():
    for obj in bpy.context.selected_objects:
        if not obj.parent and isinstance(obj.data, (bpy.types.Mesh, bpy.types.Light)):
            yield obj

def selected_meshes():
    for obj in bpy.context.selected_objects:
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def selected_objects_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
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

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
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

def normalize_scene():

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
        obj.location=obj.location*scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render mesh minimal scripts"
    )
    parser.add_argument('--obj_path', type=str)
    parser.add_argument(
        "--outdir",
        default="./render_result",
        type=str,
        help="Path to save sampled data."
    )
    
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    reset_scene()
    init_global(bpy.context)
    load_object(args.obj_path)
    normalize_scene()
    output_path = os.path.join(args.outdir, "condition.png")
    bpy.context.scene.render.filepath = (output_path)
    bpy.ops.render.render(write_still=True)
    