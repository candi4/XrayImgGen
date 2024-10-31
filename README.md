# XrayImgGen
Generate X-ray images from 3D model

## How to use
[DiffDRR](https://github.com/eigenvivek/DiffDRR)
python 3.8 can't install diffdrr. I'm using python 3.10
```shell
pip install diffdrr
```
* `.stl`: triangular mesh. We're using `mm` unit.
* `.nii`: voxel. One side of one voxel is `1mm * voxel_size`.

### Axes conventions
* In the camera coordinate system, the x-axis points downward in the image, the y-axis points leftward, and the z-axis is perpendicular to the image plane, extending outward.
