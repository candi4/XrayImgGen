# %%
import os
from tqdm import tqdm

from utils import convert_stl2nii

import yaml
with open("parameters.yaml", "r") as file:
    params = yaml.safe_load(file)
voxel_size = params["voxel_size"]

directory = params['object_directory']
object_filenames = list(set([os.path.join(directory,os.path.splitext(filename)[0]) for filename in os.listdir(directory)]))

for object_filename in (pbar := tqdm(object_filenames, desc="Convert")):
    pbar.set_postfix(object_filename=os.path.split(object_filename)[-1])
    stl_filename = object_filename + '.stl'
    nii_filename = object_filename + '.nii'
    convert_stl2nii(stl_filename=stl_filename, nii_filename=nii_filename, voxel_size=voxel_size)