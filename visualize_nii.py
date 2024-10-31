# %%
import time
start_time = time.time()

import torchio

import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np

from diffdrr.drr import DRR
from diffdrr.data import read, RigidTransform
from diffdrr.visualization import plot_drr

import yaml
with open("parameters.yaml", "r") as file:
    params = yaml.safe_load(file)
voxel_size = params["voxel_size"]


transform_matrix = torch.tensor([[1,0,0,0],
                                 [0,1,0,0],
                                 [0,0,1,300], # z < sdd
                                 [0,0,0,1]], dtype=float) # in mm
transform_matrix[:3,:3] *= voxel_size
transform = RigidTransform(transform_matrix)

# Read in the volume and get its origin and spacing in world coordinates
subject = read(torchio.ScalarImage('output.nii'),
               center_volume=False,
               orientation=None,
               transform=transform,
               )

# Initialize the DRR module for generating synthetic X-rays
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
drr = DRR(
    subject,     # An object storing the CT volume, origin, and voxel spacing
    sdd=1020.0,  # Source-to-detector distance (i.e., focal length)
    height=200,  # Image height (if width is not provided, the generated DRR is square)
    delx=0.1,    # Pixel spacing (in mm)
).to(device)

# Set the camera pose with rotations (yaw, pitch, roll) and translations (x, y, z)
rotations = torch.tensor([[0, 0, 0]], device=device)
translations = torch.tensor([[0.0, 0, 0.0]], device=device)

# 📸 Also note that DiffDRR can take many representations of SO(3) 📸
# For example, quaternions, rotation matrix, axis-angle, etc...
img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY") # shape (1,1,200,200)

# %%
image_np = img.squeeze().cpu().detach().numpy() / img.max().item()
image_np = (image_np*255).astype(np.uint8)
cv2.imwrite('output_image.png', image_np)
cv2.imshow('output_image.png', image_np); cv2.waitKey(1)

print("Consumed time:", time.time() - start_time)
# %%
