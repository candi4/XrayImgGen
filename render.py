# %%
import time
start_time = time.time()
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
from tqdm import tqdm

from utils import (object_xray, crop_nonzero, save_image, yaml_preprocessing)

import yaml
with open("parameters.yaml", "r") as file:
    params = yaml.safe_load(file)

# Prepares YAML parameters.
params = yaml_preprocessing(yalm_params=params)

voxel_size = params["voxel_size"]
object_filenames = params['object_filenames']
assembly_calibrations = params['assembly_calibrations']

for i in (pbar := tqdm(range(1000), desc="Rendering")):
    ry = random.uniform(0,360)
    rx = random.uniform(-70,70)
    delx = 0.1 # delx mm for one pixel
    image_filename=f'images/module/ry{int(ry):03}_rx{int(rx):+03}_{random.randint(0,9999):04}.png'
    pbar.set_postfix(image_filename=image_filename)

    # Transform the assembly based on the world frame
    # assembly frame based on world frame  (w_a)
    H_wc = torch.tensor(params['transform_matrix']['H_wc'], dtype=float)
    H_wc[:3,:3] = torch.tensor(R.from_euler('yx',(ry,rx), True).as_matrix())
    H_ca = torch.tensor(params['transform_matrix']['H_ca'], dtype=float)
    transform_matrix = H_wc @ H_ca
    image_nps = dict()
    for object_filename in object_filenames:
        nii_filename = object_filename + '.nii'
        # part frame based on world frame (w_i = w_a @ a_i)
        image_np = object_xray(transform_matrix=transform_matrix@assembly_calibrations[object_filename], 
                               voxel_size=voxel_size, nii_filename=nii_filename,
                               sdd=1020.0,
                               height=int(20*0.5/delx),
                               width=int(100*0.5/delx),
                               delx=delx,
                               ) # 1.06 sec for one image with delx=0.1 using GPU
        # >>> For realistic rendering >>>
        if object_filename == 'part11': # the shell-shaped part surrounding the module
            image_np /= 2
            # Add constant to non-zero pixels
            image_np[image_np != 0] += 1
        if object_filename == 'part13': # the smallest part for connecting with a wire
            pass
        if object_filename == 'part23': # the part having hooks
            image_np /= 2

        # <<< For realistic rendering <<<
        image_nps[object_filename] = image_np
    
    assembly_image = np.zeros_like(image_np)
    for object_filename in object_filenames:
        assembly_image += image_nps[object_filename]
    assembly_image = crop_nonzero(assembly_image)
    save_image(image_np=assembly_image, image_filename=image_filename, pixel_max=assembly_image.max(), printing=False)

print("Consumed time:",time.time()-start_time)
