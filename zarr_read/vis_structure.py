import zarr
pointcloud_path="/home/zh/robot_zh/3D-Diffusion-Policy/3D-Diffusion-Policy/data/adroit_hammer_expert.zarr"

pointcloud_data = zarr.open(pointcloud_path)
print(pointcloud_data.tree())