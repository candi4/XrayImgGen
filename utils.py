import torch
import numpy as np
import open3d as o3d
import nibabel as nib
import torchio
import cv2
import time
from scipy.spatial.transform import Rotation as R
import random
import os

from diffdrr.drr import DRR
from diffdrr.data import read, RigidTransform

def convert_stl2nii(stl_filename:str, nii_filename:str, voxel_size:float):
    """convert 3d mesh to 3d voxel
    """
    mesh = o3d.io.read_triangle_mesh(stl_filename)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                                voxel_size=voxel_size)
    # Convert the VoxelGrid to a numpy array
    voxels = voxel_grid.get_voxels()
    max_bound = np.asarray(voxel_grid.get_max_bound())
    min_bound = np.asarray(voxel_grid.get_min_bound())
    dim = np.ceil((max_bound - min_bound) / voxel_grid.voxel_size).astype(int)

    # Initialize the array and set values for each voxel position
    voxel_data = np.zeros(dim)

    for voxel in voxels:
        # Get the voxel grid index
        x, y, z = voxel.grid_index
        
        # Check if the coordinates are within valid range before assigning
        if 0 <= x < dim[0] and 0 <= y < dim[1] and 0 <= z < dim[2]:
            voxel_data[x, y, z] = 1

    # Save the numpy array as a NIfTI file
    nii_img = nib.Nifti1Image(voxel_data, affine=np.eye(4))
    nib.save(nii_img, nii_filename)


def object_xray(transform_matrix:torch.Tensor, voxel_size:float, nii_filename:str,
                sdd=1020.0,
                height=100,
                width=100, 
                delx=0.5,  
                ):
    """
    @param transform_matrix: part frame based on world frame (w_i)
    @param voxel_size: the value used when making nii file
    @param sdd: Source-to-detector distance (i.e., focal length)
    @param height: Image height (if width is not provided, the generated DRR is square)
    @param width
    @param delx: Pixel spacing (in mm)
    @return: image_np: It should be normalized. It may have elements larger than one.
    """
    assert transform_matrix.shape == (4,4), f"transform_matrix.shape={transform_matrix.shape}"
    transform_matrix[:3,:3] *= voxel_size
    transform = RigidTransform(transform_matrix)

    # Read in the volume and get its origin and spacing in world coordinates
    subject = read(torchio.ScalarImage(nii_filename),
                center_volume=False,
                orientation=None,
                transform=transform,
                )

    # Initialize the DRR module for generating synthetic X-rays
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drr = DRR(
        subject,     # An object storing the CT volume, origin, and voxel spacing
        sdd=sdd,
        height=height,
        width=width,
        delx=delx,   # mm for one pixel
    ).to(device)

    # Set the camera pose with rotations (yaw, pitch, roll) and translations (x, y, z)
    rotations = torch.tensor([[0, 0, 0]], device=device)
    translations = torch.tensor([[0.0, 0, 0.0]], device=device)

    # 📸 Also note that DiffDRR can take many representations of SO(3) 📸
    # For example, quaternions, rotation matrix, axis-angle, etc...
    img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY") # shape (1,1,200,200)
    image_np = img.squeeze().cpu().detach().numpy()
    return image_np

def save_image(image_np:np.ndarray, image_filename:str, pixel_max:float, imshow=False, printing=False):
    if not os.path.exists(os.path.dirname(image_filename)):
        os.makedirs(os.path.dirname(image_filename), exist_ok=True)
    if pixel_max != 0:
        image_np /= pixel_max
    image_np = (image_np*255).astype(np.uint8)
    
    cv2.imwrite(image_filename, image_np)
    if imshow: cv2.imshow(image_filename, image_np); cv2.waitKey(1)
    if printing: print(f"{image_filename} is saved")
    
def crop_nonzero(arr):
    # Find the indices of non-zero values
    nonzero_indices = np.argwhere(arr != 0)
    
    if nonzero_indices.size == 0:
        return arr  # Return the original array if there are no non-zero values
    
    # Calculate the coordinates of the rectangle
    top_left = nonzero_indices.min(axis=0)
    bottom_right = nonzero_indices.max(axis=0) + 1  # Add 1 to include the last index

    # Crop the rectangle
    return arr[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

if __name__ == "__main__":
    start_time = time.time()
    print("Start main code")

    voxel_size=0.05
    object_filenames = ['part11','part13','part23']

    make_nii = False
    if make_nii:
        print("Starting converting")
        for object_filename in object_filenames:
            stl_filename = object_filename + '.stl'
            nii_filename = object_filename + '.nii'
            convert_stl2nii(stl_filename=stl_filename, nii_filename=nii_filename, voxel_size=voxel_size)
        print("Finished converting")
    
    # The center of the larger circle of part11 is the origin of the assembly.
    # part frame based on assembly frame (a_i)
    assembly_calibrations = dict()
    assembly_calibrations['part11'] = torch.tensor([[1,0,0,-4],
                                                      [0,1,0,0],
                                                      [0,0,1,-4],
                                                      [0,0,0,1]], dtype=float)
    assembly_calibrations['part13'] = torch.tensor([[1,0,0,-3],
                                                      [0,1,0,32],
                                                      [0,0,1,-3],
                                                      [0,0,0,1]], dtype=float)
    assembly_calibrations['part23'] = torch.tensor([[0,1,0,0],
                                                    [1,0,0,0],
                                                    [0,0,1,0],
                                                    [0,0,0,1]], dtype=float) \
                                    @ torch.tensor([[1,0,0,1.683],
                                                    [0,1,0,-3.37],
                                                    [0,0,1,-3.37],
                                                    [0,0,0,1]], dtype=float)
    
    for ry in range(-90,91,10):
        for rx in range(-90,91,10):
            # Transform the assembly based on the world frame
            # assembly frame based on world frame  (w_a)
            H_wc = torch.tensor([[1,0,0,0], # in mm
                                 [0,1,0,0],
                                 [0,0,1,950], # z < sdd
                                 [0,0,0,1]], dtype=float)
            H_wc[:3,:3] = torch.tensor(R.from_euler('yx',(ry,rx), True).as_matrix())
            H_ca = torch.tensor([[1,0,0,0],
                                 [0,1,0,-45/2],
                                 [0,0,1,0],
                                 [0,0,0,1]], dtype=float)
            transform_matrix = H_wc @ H_ca
            image_nps = dict()
            for object_filename in object_filenames:
                nii_filename = object_filename + '.nii'
                delx = 0.1 # delx mm for one pixel
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

    print("Consumed time:",time.time()-start_time)
    