# Web-based Gaussian Splash Large Model Intelligent Modeling Tool

A multi-functional feedforward model for comprehensive 3D geometry prediction. It integrates multiple geometric priors (**camera pose**, **calibration intrinsics**, **depth map**) and simultaneously generates various 3D representations (**point cloud**, **multi-view depth**, **camera parameters**, **surface normals**, **3D Gaussian**) in a single forward propagation.

### Architecture

Reference: https://github.com/user-attachments/assets/ced3ef9e-8f90-423f-8ad0-ada9069111d6

## Dependencies and Installation

Option 1: One-click configuration of the runtime environment using environment.yml (Recommended)

````shell
``` Note: CUDA 12.4 is required for correct installation; please check your CUDA version.
```shell
# Check the CUDA version entered (run as administrator)
nvcc --version

# Example output
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:28:36_Pacific_Standard_Time_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0
````

Option 2: Manually install using CUDA version 12.4.

```shell
cd HunyuanWorld-Mirror
conda create -n hunyuanworld-mirror python=3.10 cmake=3.14.0 -y
conda activate hunyuanworld-mirror
conda install pytorch=2.4.0 torchvision pytorch-cuda=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
```

Should output:

```shell
NumPy: 1.26.4
Torch: 2.4.0+cu124 12.4
```

### Local Demo

```shell
# 1. Install dependencies required for the gradio demo
pip install -r requirements_demo.txt
# 2. Start the gradio demo locally
python app.py
```

### Example Code Snippet

```python
from pathlib import Path
import torch
from src.models.models.worldmirror import WorldMirror
from src.utils.inference_utils import extract_load_and_preprocess_images
# --- Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror").to(device)
# --- Load Data ---
# Load a sequence of N images into a tensor
inputs = {}
inputs['img'] = extract_load_and_preprocess_images( 
Path("path/to/your/data"), # video or directory containing images 
fps=1, # fps for extracting frames from video 
target_size=518
).to(device) # [1,N,3,H,W], in [0,1]
# -- Load Priors (Optional) --
# Configure conditioning flags and prior paths
cond_flags = [0, 0, 0] # [camera_pose, depth, intrinsics]
prior_data = { 
'camera_pose': None, # Camera pose tensor [1, N, 4, 4] 
'depthmap': None, # Depth map tensor [1, N, H, W]
'camera_intrinsics': None # Camera intrinsics tensor [1, N, 3, 3]
}
for idx, (key, data) in enumerate(prior_data.items()):
if data is not None:
cond_flags[idx] = 1
inputs[key] = data
# --- Inference ---
with torch.no_grad():
predictions = model(views=inputs, cond_flags=cond_flags)
```

<summary>Click to view output format</summary>

```python
# Geometric output
pts3d_preds, pts3d_conf = predictions["pts3d"][0], predictions["pts3d_conf"][0] # 3D point cloud in world coordinate system: [S, H, W, 3], Point cloud confidence: [S, W, H]
depth_preds, depth_conf = predictions["depth"][0], predictions["depth_conf"][0] # Z-depth in camera coordinates: [S, H, W, 1], Depth confidence: [S, W, H]
normal_preds, normal_conf = predictions["normals"][0], predictions["normals_conf"][0] # Surface normals in camera coordinates: [S, H, W, 3], Normal confidence: [S, W, H]
# Camera output
camera_poses = predictions["camera_poses"][0] # Camera pose to world (OpenCV convention): [S, 4, 4]
camera_intrs = predictions["camera_intrs"][0] # Camera intrinsic matrix: [S, 3, 3]
camera_params = predictions["camera_params"][0] # Camera vectors: [S, 9] (translation, rotation quaternions, fov_v, fov_u)
# 3D Gaussian point cloud output
splats = predictions["splats"]
means = splats["means"][0].reshape(-1, 3) # Gaussian means: [N, 3]
opacities = splats["opacities"][0].reshape(-1) # Gaussian opacities: [N]
scales = splats["scales"][0].reshape(-1, 3) # Gaussian scales: [N, 3]
quats = splats["quats"][0].reshape(-1, 4) # Gaussian quaternions: [N, 4]
sh = splats["sh"][0].reshape(-1, 1, 3) # Gaussian spherical harmonic function: [N, 1, 3]
```

Where:

- `S` is the number of input views
  
- `H, W` are the height and width of the input image
  
- `N` is the number of 3D Gaussian points
  

</details>

### More Functional Inference

For advanced usage:

- Save predictions: point cloud, depth map, normals, camera parameters, and 3D Gaussian point cloud
  
- Visualize output: depth map, surface normals, and 3D point cloud
  
- Render new views using 3D Gaussian
  
- Export 3D Gaussian point cloud results and camera parameters to COLMAP format
