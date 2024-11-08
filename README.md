# XrayImgGen
Generate X-ray images from 3D model

## How to use
python 3.8 can't install [DiffDRR](https://github.com/eigenvivek/DiffDRR). I'm using python 3.10
```shell
pip install diffdrr
```
1. Run `convert.py`. It converts `.stl` files to `.nii` files.
    * `.stl`: triangular mesh. We're using `mm` unit. (Because Inventor uses it.)
    * `.nii`: voxel. One side of one voxel is `1mm * voxel_size`.
2. Run `render.py`. It renders `.nii` files with given angles.


### Axes conventions
* In the camera coordinate system, the x-axis points downward in the image, the y-axis points leftward, and the z-axis is perpendicular to the image plane, extending outward.
