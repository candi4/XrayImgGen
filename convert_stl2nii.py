# %%
import open3d as o3d
import numpy as np
import nibabel as nib


mesh = o3d.io.read_triangle_mesh('파트11.stl')

voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=0.05)
# VoxelGrid를 numpy 배열로 변환
voxels = voxel_grid.get_voxels()
max_bound = np.asarray(voxel_grid.get_max_bound())
min_bound = np.asarray(voxel_grid.get_min_bound())
dim = np.ceil((max_bound - min_bound) / voxel_grid.voxel_size).astype(int)

# 배열 초기화 후 각 voxel 위치에 값 설정
voxel_data = np.zeros(dim)

for voxel in voxels:
    # 좌표 변환 수정
    x, y, z = (np.array(voxel.grid_index) - min_bound / voxel_grid.voxel_size).astype(int)
    
    # 유효한 범위 내의 좌표인지 확인 후 할당
    if 0 <= x < dim[0] and 0 <= y < dim[1] and 0 <= z < dim[2]:
        voxel_data[x, y, z] = 1

# numpy 배열을 NIfTI 형식으로 저장
nii_img = nib.Nifti1Image(voxel_data, affine=np.eye(4))
nib.save(nii_img, 'output.nii')