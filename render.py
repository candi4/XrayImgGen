# %%
import time
start_time = time.time()
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
from tqdm import tqdm
import os
import cv2

from utils import (object_xray, crop_nonzero, save_image, yaml_preprocessing)

import yaml
with open("parameters.yaml", "r") as file:
    params = yaml.safe_load(file)

# Prepares YAML parameters.
params = yaml_preprocessing(yalm_params=params)

voxel_size = params["voxel_size"]
object_filenames = params['object_filenames']
assembly_calibrations = params['assembly_calibrations']
directory = params['object_directory']

for i in (pbar := tqdm(range(1000), desc="Render")):
    if i == 0: rx=ry=0
    else:
        ry = 0 #random.uniform(0,360)
        rx = random.uniform(-20,20)
    delx = 0.1 # delx mm for one pixel
    image_filename=f'images/module/ry{int(ry):03}_rx{int(rx):+03}_{random.randint(0,9999):04}.png'
    pbar.set_postfix(image_filename=os.path.split(image_filename)[-1])

    # Transform the assembly based on the world frame
    # assembly frame based on world frame  (w_a)
    H_wc = torch.tensor(params['transform_matrix']['H_wc'], dtype=float)
    H_wc[:3,:3] = torch.tensor(R.from_euler('yx',(ry,rx), True).as_matrix())
    H_ca = torch.tensor(params['transform_matrix']['H_ca'], dtype=float)
    transform_matrix = H_wc @ H_ca
    image_nps = dict()
    for object_filename in object_filenames:
        nii_filename = os.path.join(directory, object_filename + '.nii')
        # part frame based on world frame (w_i = w_a @ a_i)
        image_np = object_xray(transform_matrix=transform_matrix@assembly_calibrations[object_filename], 
                               voxel_size=voxel_size, nii_filename=nii_filename,
                               sdd=1020.0,
                               height=int(20*0.5/delx),
                               width=int(100*0.5/delx),
                               delx=delx,
                               ) # 1.06 sec for one image with delx=0.1 using GPU
        # >>> For realistic rendering >>>
        contrast_medium = False
        if object_filename == 'R_part11': # the shell-shaped part surrounding the module
            if not contrast_medium:
                # 0.22 in contrast -> same as background without contrast-medium
                image_np[image_np > 0] *= 0.01
        elif object_filename == 'R_part13': # the smallest part for connecting with a wire
            if not contrast_medium:
                # Max 0.53 in contrast, Min 0.41 in contrast
                image_np /= image_np.max()
                image_np *= (0.53 - 0.41)
                image_np[image_np != 0] += 0.2
        elif object_filename == 'R_part23': # the part having hooks
            if not contrast_medium:
                # 0.4 in contrast -> 0.3 from background
                image_np[image_np != 0] = 0.4 - 0.3
        elif object_filename == 'R_part31': # Ring at the end of R_part11
            if not contrast_medium:
                # middle area is similar to the background (0.22 in constract)
                # side area is 0.40 in contract -> 0.22 from background
                image_np /= image_np.max()
                image_np[image_np > 0] = (0.40 - 0.22)*0.4
        elif object_filename in ('R_part32',  # Ring in the middle of R_part23
                                 'R_part33'): # Ring at the end    of R_part23
            if not contrast_medium:
                # 0.65 for two layers in contrast
                # 0.3 from background, 0.1 from part23
                image_np[image_np != 0] = (0.65-0.3-0.1)/2
        elif object_filename == 'R_part34': # Point at the end of R_part23
            if not contrast_medium:
                # 0.25 in paint -> 0.75 in contrast
                # 0.3 is from background.
                # 0.1 from part23
                image_np[image_np != 0] = 0.75-0.3-0.1
        elif object_filename == 'R_part41': # left  chip inside of R_part23
            if not contrast_medium:
                image_np[image_np != 0] = 0.2
        elif object_filename == 'R_part42': # right chip inside of R_part23
            if not contrast_medium:
                image_np[image_np != 0] = 0.125

        # <<< For realistic rendering <<<
        image_nps[object_filename] = image_np
    
    assembly_image = np.zeros_like(image_np)
    for object_filename in object_filenames:
        assembly_image += image_nps[object_filename]
    assembly_image = crop_nonzero(assembly_image)
    pixel_max = 1 # assembly_image.max()
    save_image(image_np=assembly_image, image_filename=image_filename, pixel_max=pixel_max, printing=False)

print("Consumed time:",time.time()-start_time)
