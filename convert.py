# %%
import time
start_time = time.time()

from utils import convert_stl2nii

import yaml
with open("parameters.yaml", "r") as file:
    params = yaml.safe_load(file)
voxel_size = params["voxel_size"]
object_filenames = params['object_filenames']

print("Starting converting")
for object_filename in object_filenames:
    cycle_start = time.time()
    stl_filename = object_filename + '.stl'
    nii_filename = object_filename + '.nii'
    convert_stl2nii(stl_filename=stl_filename, nii_filename=nii_filename, voxel_size=voxel_size)
    print("object_filename:",object_filename)
    print("    One cycle time:", time.time()-cycle_start)
print("Finished converting")
print("Consumed time:", time.time()-start_time)
