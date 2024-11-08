# %%
import time
start_time = time.time()
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils import (object_xray, crop_nonzero, save_image)

import yaml
with open("parameters.yaml", "r") as file:
    params = yaml.safe_load(file)
voxel_size = params["voxel_size"]
object_filenames = params['object_filenames']
assembly_calibrations = params['assembly_calibrations']

# The center of the larger circle of part11 is the origin of the assembly.
# part frame based on assembly frame (a_i)
for part, matrix in assembly_calibrations.items():
    matrix = torch.tensor(matrix, dtype=float)
    if matrix.shape == (4,4):
        assembly_calibrations[part] = torch.tensor(matrix, dtype=float)
    else:
        assembly_calibrations[part] = torch.eye(4, dtype=float)
        for x in matrix:
            x = torch.tensor(x, dtype=float)
            assembly_calibrations[part] = assembly_calibrations[part] @ x
assert set(object_filenames).issubset(set(assembly_calibrations.keys())), f"assembly_calibrations={assembly_calibrations}\nobject_filenames={object_filenames}"


for ry in range(-90,91,10):
    for rx in range(-90,91,10):
        delx = 0.1 # delx mm for one pixel
        cycle_start = time.time()
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
                                    delx=delx,) # 1.06 sec for one image with delx=0.1 using GPU
            image_nps[object_filename] = image_np
        
        assembly_image = np.zeros_like(image_np)
        for object_filename in object_filenames:
            assembly_image += image_nps[object_filename]
        assembly_image = crop_nonzero(assembly_image)
        save_image(image_np=assembly_image, image_filename=f'images/module/(ry{ry})(rx{rx})(delx{delx}).png', pixel_max=assembly_image.max(), printing=True)
        print("    One cycle time:", time.time()-cycle_start)

print("Consumed time:",time.time()-start_time)
