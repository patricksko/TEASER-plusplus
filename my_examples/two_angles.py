import numpy as np
import pyrender
import trimesh
import cv2
import os
def look_at(eye, target, up=np.array([0, 1, 0])):
    forward = np.array(target) - np.array(eye)
    forward /= np.linalg.norm(forward)

    if np.abs(np.dot(forward, up)) > 0.999:
        up = np.array([0, 0, 1])

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    true_up = np.cross(forward, right)
    true_up /= np.linalg.norm(true_up)

    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = true_up
    mat[:3, 2] = -forward  # NEGATIVE forward to look along -Z
    mat[:3, 3] = eye
    return mat


def get_front_and_left_views(radius=0.2, angle_deg=5):
    """
    Returns two camera positions:
    - One directly in front of the object
    - One slightly rotated to the left around the Y-axis
    """
    directions = []

    # Front view (along +Z)
    front = np.array([radius, 0, 0])
    directions.append(front)

    # Slightly from the left (rotate around Y-axis)
    angle_rad = np.deg2rad(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    left_view = rot_matrix @ front
    directions.append(left_view)

    return directions
# Load the 3D model
mesh = trimesh.load("/home/skoumal/dev/BlenderProc/LegoBlock_variant.ply")
blue_color = np.array([0, 0, 255, 255], dtype=np.uint8)  # RGBA
mesh.visual.vertex_colors = np.tile([0, 0, 255, 255], (len(mesh.vertices), 1))  # solid blue

material = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[0, 0, 1, 1],  # solid blue, fully opaque
    metallicFactor=0.0,
    roughnessFactor=1.0,
    alphaMode='OPAQUE',
    doubleSided=True    # important: disables backface culling, makes mesh look solid from all sides
)

pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

scene = pyrender.Scene()
#mesh = pyrender.Mesh.from_trimesh(mesh)
scene.add(pyrender_mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)

# Set up the renderer
renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

# Define camera intrinsics
focal_length = 500.0
camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=320, cy=240)
cam_node = scene.add(camera)

# Add a directional light
light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
scene.add(light)

# Output folder
output_dir = "lego_rot_views"
os.makedirs(output_dir, exist_ok=True)

directions = get_front_and_left_views(radius=0.2)
target = np.array([0, 0, 0])

for i, eye in enumerate(directions):
    cam_pose = look_at(eye, target)
    scene.set_pose(cam_node, pose=cam_pose)
    color, depth = renderer.render(scene)

    # Save RGB image
    cv2.imwrite(os.path.join(output_dir, f"rgb_{i:02d}.png"), color)

    # Save depth
    np.save(os.path.join(output_dir, f"depth_{i:02d}.npy"), depth)

    # Save pose
    np.save(os.path.join(output_dir, f"pose_{i:02d}.npy"), cam_pose)

    # Generate and save segmentation mask
    mask = (depth > 0).astype(np.uint8) * 255  # 0 = bg, 255 = object
    cv2.imwrite(os.path.join(output_dir, f"mask_{i:02d}.png"), mask)



