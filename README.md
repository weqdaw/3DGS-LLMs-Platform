# 网页端高斯泼溅大模型智能建模工具

一个多功能的前馈模型，用于全面的3D几何预测。它整合了多种几何先验（**相机位姿**、**校准内参**、**深度图**），并在单次前向传播中同时生成各种3D表示（**点云**、**多视图深度**、**相机参数**、**表面法线**、**3D高斯**）。

### 架构

参考：https://github.com/user-attachments/assets/ced3ef9e-8f90-423f-8ad0-ada9069111d6

## 依赖和安装
方案1：用environment.yml一键配置运行环境（推荐）
```shell
conda env create -f environment.yml
```
注意：CUDA 12.4 才能正确安装，请检查CUDA版本。
```shell
# 检查输入的CUDA版本（管理员下运行）
nvcc --version
# 示例输出
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:28:36_Pacific_Standard_Time_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0
```

方案2：使用 CUDA 12.4 版本进行手动安装。
```shell
cd HunyuanWorld-Mirror
conda create -n hunyuanworld-mirror python=3.10 cmake=3.14.0 -y
conda activate hunyuanworld-mirror
conda install pytorch=2.4.0 torchvision pytorch-cuda=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124
```
如果发现问题，请确保检查输出与示例保持一致

```shell
python -c "import numpy, torch; print('NumPy:', numpy.__version__); print('Torch:', torch.__version__, torch.version.cuda)"
```

应输出：
```shell
NumPy: 1.26.4
Torch: 2.4.0+cu124 12.4
```
### 本地演示

```shell
# 1. 安装 gradio 演示所需的依赖
pip install -r requirements_demo.txt
# 2. 在本地启动 gradio 演示
python app.py
```

### 示例代码片段

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
    fps=1, # fps for extracing frames from video
    target_size=518
).to(device)  # [1,N,3,H,W], in [0,1]

# -- Load Priors (Optional) --
# Configure conditioning flags and prior paths
cond_flags = [0, 0, 0]  # [camera_pose, depth, intrinsics]
prior_data = {
    'camera_pose': None,      # Camera pose tensor [1, N, 4, 4]
    'depthmap': None,         # Depth map tensor [1, N, H, W]
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

<details>
<summary>点击查看输出格式</summary>

```python
# 几何输出
pts3d_preds, pts3d_conf = predictions["pts3d"][0], predictions["pts3d_conf"][0]      # 世界坐标系中的3D点云：[S, H, W, 3], 点云置信度: [S, W, H] 
depth_preds, depth_conf = predictions["depth"][0], predictions["depth_conf"][0]      # 相机坐标系中的Z深度：[S, H, W, 1], 深度置信度: [S, W, H] 
normal_preds, normal_conf = predictions["normals"][0], predictions["normals_conf"][0] # 相机坐标系中的表面法线：[S, H, W, 3], 法线置信度: [S, W, H] 

# 相机输出
camera_poses = predictions["camera_poses"][0]  # 相机到世界的位姿（OpenCV约定）：[S, 4, 4]
camera_intrs = predictions["camera_intrs"][0]  # 相机内参矩阵：[S, 3, 3]
camera_params = predictions["camera_params"][0]   # 相机向量：[S, 9]（平移，旋转四元数，fov_v，fov_u）

# 3D 高斯点云输出
splats = predictions["splats"]
means = splats["means"][0].reshape(-1, 3)      # 高斯均值：[N, 3]
opacities = splats["opacities"][0].reshape(-1) # 高斯不透明度：[N]
scales = splats["scales"][0].reshape(-1, 3)    # 高斯尺度：[N, 3]
quats = splats["quats"][0].reshape(-1, 4)      # 高斯四元数：[N, 4]
sh = splats["sh"][0].reshape(-1, 1, 3)         # 高斯球谐函数：[N, 1, 3]
```

其中：

- `S` 是输入视图的数量
- `H, W` 是输入图像的高度和宽度
- `N` 是3D高斯的数量

</details>

### 更多功能的推理

对于高级用法:

- 保存预测：点云、深度图、法线、相机参数和3D高斯点云
- 可视化输出：深度图、表面法线和3D点云
- 使用3D高斯渲染新视图
- 将3D高斯点云结果和相机参数导出为 COLMAP 格式

## 后期3DGS 优化

output/
├── images/                 # 输入图像
├── sparse/
│   └── 0/
│       ├── cameras.bin     # 相机内参
│       ├── images.bin      # 相机位姿
│       └── points3D.bin    # 3D点
└── gaussians.ply           # 3D高斯点云初始化
```
