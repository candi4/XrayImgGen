import torch
import numpy as np
import open3d as o3d
import nibabel as nib
import torchio
import cv2
import time

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


def object_xray(transform_matrix:torch.Tensor, voxel_size:float,
                sdd=1020.0,
                height=100,
                width=100, 
                delx=0.5,  
                ):
    """
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
    subject = read(torchio.ScalarImage('output.nii'),
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
        delx=delx,
    ).to(device)

    # Set the camera pose with rotations (yaw, pitch, roll) and translations (x, y, z)
    rotations = torch.tensor([[0, 0, 0]], device=device)
    translations = torch.tensor([[0.0, 0, 0.0]], device=device)

    # 📸 Also note that DiffDRR can take many representations of SO(3) 📸
    # For example, quaternions, rotation matrix, axis-angle, etc...
    img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY") # shape (1,1,200,200)
    image_np = img.squeeze().cpu().detach().numpy()
    return image_np

def save_image(image_np:np.ndarray, image_filename:str, pixel_max:float):
    image_np /= pixel_max
    image_np = (image_np*255).astype(np.uint8)
    
    cv2.imwrite(image_filename, image_np)
    cv2.imshow(image_filename, image_np); cv2.waitKey(1)
    

if __name__ == "__main__":
    start_time = time.time()
    print("Start main code")

    stl_filename='파트11.stl'
    nii_filename='output.nii'
    voxel_size=0.05
    convert_stl2nii(stl_filename=stl_filename, nii_filename=nii_filename, voxel_size=voxel_size)


    transform_matrix = torch.tensor([[1,0,0,0],
                                     [0,1,0,0],
                                     [0,0,1,300], # z < sdd
                                     [0,0,0,1]], dtype=float) # in mm
    image_np = object_xray(transform_matrix=transform_matrix, voxel_size=voxel_size,
                            sdd=1020.0,
                            height=100, 
                            width=100, 
                            delx=0.5,)
    
    image_filename = 'output_image.png'
    pixel_max = image_np.max()
    save_image(image_np, image_filename, pixel_max)
    
    print("Consumed time:",time.time()-start_time)