import gc
import os
import shutil
import time
from datetime import datetime
import io
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image
from pillow_heif import register_heif_opener
register_heif_opener()

from src.utils.inference_utils import load_and_preprocess_images
from src.utils.geometry import (
    depth_edge,
    normals_edge
)
from src.utils.visual_util import (
    convert_predictions_to_glb_scene,
    segment_sky,
    download_file_from_url
)
from src.utils.save_utils import save_camera_params, save_gs_ply, process_ply_to_splat, convert_gs_to_ply, save_scene_ply, save_points_ply
from src.utils.render_utils import render_interpolated_video
import onnxruntime


# Initialize model - this will be done on GPU when needed
model = None

# Global variable to store current terminal output
current_terminal_output = ""

# Helper class to capture terminal output
class TeeOutput:
    """捕获输出同时仍打印到控制台"""
    def __init__(self, max_chars=10000):
        self.terminal = sys.stdout
        self.log = io.StringIO()
        self.max_chars = max_chars  # 限制最大字符数
    
    def write(self, message):
        global current_terminal_output
        self.terminal.write(message)
        self.log.write(message)
        
        # 获取当前内容并限制长度
        content = self.log.getvalue()
        if len(content) > self.max_chars:
            # 只保留最后 max_chars 个字符
            content = "...(earlier output truncated)...\n" + content[-self.max_chars:]
            self.log = io.StringIO()
            self.log.write(content)
        
        current_terminal_output = self.log.getvalue()
    
    def flush(self):
        self.terminal.flush()
    
    def getvalue(self):
        return self.log.getvalue()
    
    def clear(self):
        global current_terminal_output
        self.log = io.StringIO()
        current_terminal_output = ""

def create_confidence_mask(confidence: torch.Tensor,
                          conf_threshold_percent: float = 30.0) -> torch.Tensor:
    """
    创建基于置信度阈值的过滤掩码
    丢弃底部p%的置信度点，保留顶部(100-p)%
    
    Args:
        confidence: 置信度分数 (任意形状)
        conf_threshold_percent: 要过滤的低置信度点百分比 (0-100)
    
    Returns:
        用于过滤点的布尔掩码
    """
    # 展平置信度分数
    conf_flat = confidence.flatten()
    # 屏蔽极小/无效值
    conf_flat = conf_flat.masked_fill(conf_flat <= 1e-5, float("-inf"))
    
    N = conf_flat.numel()
    
    # 丢弃底部p%，保留顶部(100-p)%
    if conf_threshold_percent > 0:
        keep_from_percent = int(np.ceil(N * (100.0 - conf_threshold_percent) / 100.0))
    else:
        keep_from_percent = N
    K = max(1, keep_from_percent)
    
    # 选择top-K索引 (确定性，无随机性)
    topk_idx = torch.topk(conf_flat, K, largest=True, sorted=False).indices
    
    # 创建掩码
    conf_mask = torch.zeros_like(conf_flat, dtype=torch.bool)
    conf_mask[topk_idx] = True
    
    return conf_mask

# -------------------------------------------------------------------------
# Model inference
# -------------------------------------------------------------------------
@spaces.GPU(duration=120)
def run_model(
    target_dir,
    confidence_percentile: float = 10,
    edge_normal_threshold: float = 5.0,
    edge_depth_threshold: float = 0.03,
    apply_confidence_mask: bool = True,
    apply_edge_mask: bool = True,
    conf_threshold: float = 0.0,
    target_size: int = 518,
    save_depth_maps: bool = True,
    save_normal_maps: bool = True,
    save_point_cloud: bool = True,
    save_gaussians: bool = True,
    save_colmap: bool = False,
    save_rendered_video: bool = True,
    enable_amp: bool = True,
    cond_pose: bool = False,
    cond_intrinsics: bool = False,
    cond_depth: bool = False,
):
    """
    在 'target_dir/images' 文件夹中的图像上运行 WorldMirror 模型并返回预测结果。
    """
    global model
    import torch  # Ensure torch is available in function scope
    
    from src.models.models.worldmirror import WorldMirror
    from src.models.utils.geometry import depth_to_world_coords_points

    print(f"正在处理来自 {target_dir} 的图像")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Initialize model if not already done
    if model is None:
        model = WorldMirror.from_pretrained("tencent/HunyuanWorld-Mirror").to(device)
    else:
        model.to(device)
    
    model.eval()

    # Load images using WorldMirror's load_images function
    print("正在加载图像...")
    image_folder_path = os.path.join(target_dir, "images")
    image_file_paths = [os.path.join(image_folder_path, path) for path in os.listdir(image_folder_path)]
    img = load_and_preprocess_images(image_file_paths).to(device)

    print(f"已加载 {img.shape[1]} 张图像")
    if img.shape[1] == 0:
        raise ValueError("未找到图像。请检查您的上传。")

    # Run model inference
    print("正在运行推理...")
    inputs = {}
    inputs['img'] = img
    
    # 设置条件化标志
    cond_flags = [0, 0, 0]  # [camera_pose, depth, intrinsics]
    if cond_pose:
        cond_flags[0] = 1
    if cond_depth:
        cond_flags[1] = 1
    if cond_intrinsics:
        cond_flags[2] = 1
    
    # 混合精度设置
    use_amp = enable_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_amp:
        amp_dtype = torch.bfloat16
        print("启用混合精度推理 (bfloat16)")
    else:
        amp_dtype = torch.float32
        print("使用标准精度推理 (float32)")
    
    with torch.amp.autocast('cuda', enabled=bool(use_amp), dtype=amp_dtype):
        predictions = model(views=inputs, cond_flags=cond_flags)

    # img
    imgs = inputs["img"].permute(0, 1, 3, 4, 2)
    imgs = imgs[0].detach().cpu().numpy() # S H W 3

    # depth output
    depth_preds = predictions["depth"]
    depth_conf = predictions["depth_conf"]
    depth_preds = depth_preds[0].detach().cpu().numpy() # S H W 1
    depth_conf = depth_conf[0].detach().cpu().numpy() # S H W

    # normal output
    normal_preds = predictions["normals"] # S H W 3
    normal_preds = normal_preds[0].detach().cpu().numpy() # S H W 3

    # camera parameters
    camera_poses = predictions["camera_poses"][0].detach().cpu().numpy() # [S,4,4]
    camera_intrs = predictions["camera_intrs"][0].detach().cpu().numpy() # [S,3,3]
    
    # points output
    pts3d_preds = depth_to_world_coords_points(predictions["depth"][0, ..., 0], predictions["camera_poses"][0], predictions["camera_intrs"][0])[0]
    pts3d_preds = pts3d_preds.detach().cpu().numpy()  # S H W 3
    pts3d_conf = depth_conf              # S H W

    # sky mask segmentation
    if not os.path.exists("skyseg.onnx"):
        print("正在下载 skyseg.onnx...")
        download_file_from_url(
            "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx"
        )
    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    sky_mask_list = []
    for i, img_path in enumerate([os.path.join(image_folder_path, path) for path in os.listdir(image_folder_path)]):
        sky_mask = segment_sky(img_path, skyseg_session)
        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != imgs.shape[1] or sky_mask.shape[1] != imgs.shape[2]:
            sky_mask = cv2.resize(sky_mask, (imgs.shape[2], imgs.shape[1]))
        sky_mask_list.append(sky_mask)
    sky_mask = np.stack(sky_mask_list, axis=0) # [S, H, W]
    sky_mask = sky_mask>0

    # mask computation
    final_mask_list = []    
    for i in range(inputs["img"].shape[1]):
        final_mask = None
        if apply_confidence_mask:
            # compute confidence mask based on the pointmap confidence
            confidences = pts3d_conf[i, :, :] # [H, W]
            percentile_threshold = np.quantile(confidences, confidence_percentile / 100.0)
            conf_mask = confidences >= percentile_threshold
            if final_mask is None:
                final_mask = conf_mask
            else:
                final_mask = final_mask & conf_mask
        if apply_edge_mask:
            # compute edge mask based on the normalmap
            normal_pred = normal_preds[i] # [H, W, 3]
            normal_edges = normals_edge(
                normal_pred, tol=edge_normal_threshold, mask=final_mask
            )
            # compute depth mask based on the depthmap
            depth_pred = depth_preds[i, :, :, 0] # [H, W]
            depth_edges = depth_edge(
                depth_pred, rtol=edge_depth_threshold, mask=final_mask
            )
            edge_mask = ~(depth_edges & normal_edges)
            if final_mask is None:
                final_mask = edge_mask
            else:
                final_mask = final_mask & edge_mask
        final_mask_list.append(final_mask)

    if final_mask_list[0] is not None:
        final_mask = np.stack(final_mask_list, axis=0) # [S, H, W]
    else:
        final_mask = np.ones(pts3d_conf.shape[:3], dtype=bool) # [S, H, W]

    # gaussian splatting output
    if "splats" in predictions:
        splats_dict = {}
        splats_dict['means'] = predictions["splats"]["means"]
        splats_dict['scales'] = predictions["splats"]["scales"]
        splats_dict['quats'] = predictions["splats"]["quats"]
        splats_dict['opacities'] = predictions["splats"]["opacities"]
        if "sh" in predictions["splats"]:
            splats_dict['sh'] = predictions["splats"]["sh"]
        if "colors" in predictions["splats"]:
            splats_dict['colors'] = predictions["splats"]["colors"]

    # output lists
    outputs = {}
    outputs['images'] = imgs
    outputs['world_points'] = pts3d_preds
    outputs['depth'] = depth_preds
    outputs['normal'] = normal_preds
    outputs['final_mask'] = final_mask
    outputs['sky_mask'] = sky_mask
    outputs['camera_poses'] = camera_poses
    outputs['camera_intrs'] = camera_intrs
    if "splats" in predictions:
        outputs['splats'] = splats_dict
    
    # Process data for visualization tabs (depth, normal)
    processed_data = prepare_visualization_data(
        outputs, inputs
    )

    # 高级保存功能
    print("正在保存高级功能输出...")
    
    # 保存深度图
    if save_depth_maps:
        depth_dir = os.path.join(target_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        for i in range(depth_preds.shape[0]):
            depth_png_path = os.path.join(depth_dir, f"depth_{i:04d}.png")
            depth_npy_path = os.path.join(depth_dir, f"depth_{i:04d}.npy")
            
            # 保存PNG可视化
            depth_vis = render_depth_visualization(depth_preds[i].squeeze(), final_mask[i])
            if depth_vis is not None:
                Image.fromarray(depth_vis).save(depth_png_path)
            
            # 保存原始深度数据
            np.save(depth_npy_path, depth_preds[i].squeeze())
        print(f"  - 保存了 {depth_preds.shape[0]} 个深度图到 {depth_dir}")
    
    # 保存法线图
    if save_normal_maps:
        normal_dir = os.path.join(target_dir, "normal")
        os.makedirs(normal_dir, exist_ok=True)
        for i in range(normal_preds.shape[0]):
            normal_png_path = os.path.join(normal_dir, f"normal_{i:04d}.png")
            
            # 保存法线可视化
            normal_vis = render_normal_visualization(normal_preds[i], final_mask[i])
            if normal_vis is not None:
                Image.fromarray(normal_vis).save(normal_png_path)
        print(f"  - 保存了 {normal_preds.shape[0]} 个法线图到 {normal_dir}")
    
    # 保存点云PLY
    if save_point_cloud and conf_threshold > 0:
        pts_list = []
        pts_colors_list = []
        pts_conf_list = []
        
        for i in range(pts3d_preds.shape[0]):
            pts = torch.from_numpy(pts3d_preds[i])  # [H,W,3]
            pts_conf = torch.from_numpy(pts3d_conf[i])  # [H,W]
            img_colors = torch.from_numpy(imgs[i])  # [H, W, 3]
            img_colors = (img_colors * 255).to(torch.uint8)
            
            pts_list.append(pts.reshape(-1, 3))
            pts_colors_list.append(img_colors.reshape(-1, 3))
            pts_conf_list.append(pts_conf.reshape(-1))

        all_pts = torch.cat(pts_list, dim=0)
        all_colors = torch.cat(pts_colors_list, dim=0)
        all_conf = torch.cat(pts_conf_list, dim=0)
        
        # 应用置信度过滤
        conf_mask = create_confidence_mask(all_conf, conf_threshold)
        filtered_pts = all_pts[conf_mask]
        filtered_colors = all_colors[conf_mask]
        
        pts_ply_path = os.path.join(target_dir, "filtered_points.ply")
        save_scene_ply(pts_ply_path, filtered_pts, filtered_colors)
        print(f"  - 保存了 {len(filtered_pts)} 个过滤后的点到 {pts_ply_path}")
    
    # 保存COLMAP格式
    if save_colmap:
        try:
            from src.utils.build_pycolmap_recon import build_pycolmap_reconstruction
            from src.models.utils.camera_utils import vector_to_camera_matrices
            from src.models.utils.geometry import create_pixel_coordinate_grid
            
            sparse_dir = os.path.join(target_dir, "sparse", "0")
            os.makedirs(sparse_dir, exist_ok=True)
            
            # 准备相机参数
            H, W = depth_preds.shape[1], depth_preds.shape[2]
            camera_params = predictions["camera_params"][0] if "camera_params" in predictions else None
            
            if camera_params is not None:
                e3x4, intr = vector_to_camera_matrices(
                    torch.from_numpy(camera_params), 
                    image_hw=(H, W)
                )
                extrinsics = e3x4[0].detach().cpu().numpy()  # [S,3,4]
                intrinsics = intr[0].detach().cpu().numpy()  # [S,3,3]
                
                # 构建点云数据
                points_list = []
                colors_list = []
                conf_list = []
                xyf_list = []
                
                xyf_grid = create_pixel_coordinate_grid(
                    num_frames=depth_preds.shape[0], 
                    height=H, 
                    width=W
                ).astype(np.int32)
                
                for i in range(depth_preds.shape[0]):
                    d = torch.from_numpy(depth_preds[i, :, :, 0])
                    d_conf = torch.from_numpy(pts3d_conf[i])
                    c2w = torch.from_numpy(camera_poses[i][:3, :4])  # [3, 4]
                    K = torch.from_numpy(camera_intrs[i])
                    
                    pts_i, _, mask = depth_to_world_coords_points(
                        d[None], c2w[None], K[None]
                    )
                    
                    img_colors = torch.from_numpy(imgs[i]) * 255
                    img_colors = img_colors.to(torch.uint8)
                    valid = mask[0]
                    
                    if valid.sum().item() > 0:
                        xyf_np = xyf_grid[i][valid.cpu().numpy()]
                        xyf_list.append(torch.from_numpy(xyf_np).to(valid.device))
                        points_list.append(pts_i[0][valid])
                        colors_list.append(img_colors[valid])
                        conf_list.append(d_conf[valid])

                if points_list:
                    all_pts = torch.cat(points_list, dim=0)
                    all_cols = torch.cat(colors_list, dim=0)
                    all_conf = torch.cat(conf_list, dim=0)
                    all_xyf = torch.cat(xyf_list, dim=0)

                    # 全局置信度过滤
                    conf_mask = create_confidence_mask(all_conf, conf_threshold)
                    
                    # 转换为numpy
                    f_pts = all_pts[conf_mask].detach().cpu().numpy()
                    f_cols = all_cols[conf_mask].detach().cpu().numpy()
                    f_xyf = all_xyf[conf_mask].detach().cpu().numpy()
                    
                    # 构建COLMAP重建
                    image_size = np.array([W, H])
                    reconstruction = build_pycolmap_reconstruction(
                        points=f_pts,
                        pixel_coords=f_xyf,
                        point_colors=f_cols,
                        poses=extrinsics,
                        intrinsics=intrinsics,
                        image_size=image_size,
                        shared_camera_model=False,
                        camera_model="SIMPLE_PINHOLE",
                    )
                    
                    # 更新图像名称
                    image_names = [f"image_{i+1:04d}.png" for i in range(len(extrinsics))]
                    for pyimageid in reconstruction.images:
                        reconstruction.images[pyimageid].name = image_names[pyimageid - 1]
                    
                    # 写入COLMAP文件
                    reconstruction.write(sparse_dir)
                    
                    # 保存points3D.ply
                    from src.utils.save_utils import save_points_ply
                    save_points_ply(os.path.join(sparse_dir, "points3D.ply"), f_pts, f_cols)
                    
                    print(f"  - 保存了COLMAP格式到 {sparse_dir}")
                else:
                    print("  - 警告：没有有效的点云数据用于COLMAP导出")
            else:
                print("  - 警告：缺少相机参数，无法导出COLMAP格式")
        except Exception as e:
            print(f"  - COLMAP导出失败: {e}")

    # Clean up
    torch.cuda.empty_cache()

    return outputs, processed_data


# -------------------------------------------------------------------------
# Update and navigation function
# -------------------------------------------------------------------------
def update_view_info(current_view, total_views, view_type="深度"):
        """更新视图信息显示"""
        return f"""
        <div style='text-align: center; padding: 10px; background: #f8f8f8; color: #999; border-radius: 8px; margin-bottom: 10px;'>
            <strong>{view_type} 视图导航</strong> | 
            当前: 视图 {current_view} / {total_views} 视图
        </div>
        """
        
def update_view_selectors(processed_data):
    """根据可用视图更新视图选择滑块和信息显示"""
    if processed_data is None or len(processed_data) == 0:
        num_views = 1
    else:
        num_views = len(processed_data)

    # 确保 num_views 至少为 1
    num_views = max(1, num_views)

    # 更新滑块的最大值和视图信息，使用 gr.update() 而不是创建新组件
    depth_slider_update = gr.update(minimum=1, maximum=num_views, value=1, step=1)
    normal_slider_update = gr.update(minimum=1, maximum=num_views, value=1, step=1)
    
    # 更新视图信息显示
    depth_info_update = update_view_info(1, num_views, "深度")
    normal_info_update = update_view_info(1, num_views, "法线")

    return (
        depth_slider_update,  # depth_view_slider
        normal_slider_update,  # normal_view_slider
        depth_info_update,    # depth_view_info
        normal_info_update,   # normal_view_info
    )

def get_view_data_by_index(processed_data, view_index):
    """通过索引获取视图数据，处理边界"""
    if processed_data is None or len(processed_data) == 0:
        return None

    view_keys = list(processed_data.keys())
    if view_index < 0 or view_index >= len(view_keys):
        view_index = 0

    return processed_data[view_keys[view_index]]

def update_depth_view(processed_data, view_index):
    """更新特定视图索引的深度视图"""
    view_data = get_view_data_by_index(processed_data, view_index)
    if view_data is None or view_data["depth"] is None:
        return None

    return render_depth_visualization(view_data["depth"], mask=view_data.get("mask"))

def update_normal_view(processed_data, view_index):
    """更新特定视图索引的法线视图"""
    view_data = get_view_data_by_index(processed_data, view_index)
    if view_data is None or view_data["normal"] is None:
        return None

    return render_normal_visualization(view_data["normal"], mask=view_data.get("mask"))

def initialize_depth_normal_views(processed_data):
    """使用第一个视图数据初始化深度和法线视图显示"""
    if processed_data is None or len(processed_data) == 0:
        return None, None

    # Use update functions to ensure confidence filtering is applied from the start
    depth_vis = update_depth_view(processed_data, 0)
    normal_vis = update_normal_view(processed_data, 0)

    return depth_vis, normal_vis


# -------------------------------------------------------------------------
# File upload and update preview gallery
# -------------------------------------------------------------------------
def process_uploaded_files(files, time_interval=1.0):
    """
    通过提取视频帧或复制图像来处理上传的文件。
    
    Args:
        files: 上传的文件对象列表（视频或图像）
        time_interval: 视频帧提取的间隔（秒）
        
    Returns:
        tuple: (target_dir, image_paths) 其中 target_dir 是输出目录
               image_paths 是处理后的图像文件路径列表
    """
    gc.collect()
    torch.cuda.empty_cache()

    # Create unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    input_base_dir = "input"
    if not os.path.exists(input_base_dir):
        os.makedirs(input_base_dir)

    target_dir = os.path.join(input_base_dir,f"input_images_{timestamp}")
    images_dir = os.path.join(target_dir, "images")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(images_dir)

    image_paths = []

    if files is None:
        return target_dir, image_paths

    video_exts = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".3gp"]

    for file_data in files:
        # Get file path
        if isinstance(file_data, dict) and "name" in file_data:
            src_path = file_data["name"]
        else:
            src_path = str(file_data)

        ext = os.path.splitext(src_path)[1].lower()
        base_name = os.path.splitext(os.path.basename(src_path))[0]

        # Process video: extract frames
        if ext in video_exts:
            cap = cv2.VideoCapture(src_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = int(fps * time_interval)

            frame_count = 0
            saved_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % interval == 0:
                    dst_path = os.path.join(images_dir, f"{base_name}_{saved_count:06}.png")
                    cv2.imwrite(dst_path, frame)
                    image_paths.append(dst_path)
                    saved_count += 1
            cap.release()
            print(f"从 {os.path.basename(src_path)} 提取了 {saved_count} 帧")

        # Process HEIC/HEIF: convert to JPEG
        elif ext in [".heic", ".heif"]:
            try:
                with Image.open(src_path) as img:
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    dst_path = os.path.join(images_dir, f"{base_name}.jpg")
                    img.save(dst_path, "JPEG", quality=95)
                    image_paths.append(dst_path)
                    print(f"转换 HEIC: {os.path.basename(src_path)} -> {os.path.basename(dst_path)}")
            except Exception as e:
                print(f"HEIC 转换失败 {src_path}: {e}")
                dst_path = os.path.join(images_dir, os.path.basename(src_path))
                shutil.copy(src_path, dst_path)
                image_paths.append(dst_path)

        # Process regular images: copy directly
        else:
            dst_path = os.path.join(images_dir, os.path.basename(src_path))
            shutil.copy(src_path, dst_path)
            image_paths.append(dst_path)

    image_paths = sorted(image_paths)

    print(f"文件已处理到 {images_dir}")
    return target_dir, image_paths

# Handle file upload and update preview gallery
def update_gallery_on_upload(input_video, input_images, time_interval=1.0):
    """
    当用户上传或更改文件时立即处理上传的文件，
    并在图库中显示它们。返回 (target_dir, image_paths)。
    如果没有上传任何内容，返回 None 和空列表。
    """
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = process_uploaded_files(input_video, input_images, time_interval)
    return (
        None,
        target_dir,
        image_paths,
        "上传完成。点击'重建'开始3D处理。",
    )
        
# -------------------------------------------------------------------------
# Init function
# -------------------------------------------------------------------------
def prepare_visualization_data(
    model_outputs, input_views
):
    """将模型预测转换为结构化格式以供显示组件使用"""
    visualization_dict = {}

    # Iterate through each input view
    nviews = input_views["img"].shape[1]
    for idx in range(nviews):
        # Extract RGB image data
        rgb_image = input_views["img"][0, idx].detach().cpu().numpy()

        # Retrieve 3D coordinate predictions
        world_coordinates = model_outputs["world_points"][idx]

        # Build view-specific data structure
        current_view_info = {
            "image": rgb_image,
            "points3d": world_coordinates,
            "depth": None,
            "normal": None,
            "mask": None,
        }

        # Apply final segmentation mask from model
        segmentation_mask = model_outputs["final_mask"][idx].copy()

        current_view_info["mask"] = segmentation_mask
        current_view_info["depth"] = model_outputs["depth"][idx].squeeze()

        surface_normals = model_outputs["normal"][idx]
        current_view_info["normal"] = surface_normals

        visualization_dict[idx] = current_view_info

    return visualization_dict

@spaces.GPU(duration=120)
def gradio_demo(
    target_dir,
    frame_selector="全部",
    show_camera=False,
    filter_sky_bg=False,
    show_mesh=False,
    filter_ambiguous=False,
    conf_threshold=0.0,
    target_size=518,
    save_depth_maps=True,
    save_normal_maps=True,
    save_point_cloud=True,
    save_gaussians=True,
    save_colmap=False,
    save_rendered_video=True,
    enable_amp=True,
    cond_pose=False,
    cond_intrinsics=False,
    cond_depth=False,
):
    """
    使用已创建的 target_dir/images 执行重建。
    """
    # Capture terminal output
    tee = TeeOutput()
    old_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        if not os.path.isdir(target_dir) or target_dir == "None":
            terminal_log = tee.getvalue()
            sys.stdout = old_stdout
            return None, "未找到有效的目标目录。请先上传。", None, None, None, None, None, None, None, None, None, None, None, None, terminal_log

        start_time = time.time()
        gc.collect()
        torch.cuda.empty_cache()

        # Prepare frame_selector dropdown
        target_dir_images = os.path.join(target_dir, "images")
        all_files = (
            sorted(os.listdir(target_dir_images))
            if os.path.isdir(target_dir_images)
            else []
        )
        all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
        frame_selector_choices = ["全部"] + all_files

        print("正在运行 WorldMirror 模型...")
        with torch.no_grad():
            predictions, processed_data = run_model(
                target_dir,
                conf_threshold=conf_threshold,
                target_size=target_size,
                save_depth_maps=save_depth_maps,
                save_normal_maps=save_normal_maps,
                save_point_cloud=save_point_cloud,
                save_gaussians=save_gaussians,
                save_colmap=save_colmap,
                save_rendered_video=save_rendered_video,
                enable_amp=enable_amp,
                cond_pose=cond_pose,
                cond_intrinsics=cond_intrinsics,
                cond_depth=cond_depth,
            )

        # Save predictions
        prediction_save_path = os.path.join(target_dir, "predictions.npz")
        np.savez(prediction_save_path, **predictions)

        # Save camera parameters as JSON
        camera_params_file = save_camera_params(
            predictions['camera_poses'], 
            predictions['camera_intrs'], 
            target_dir
        )

        # Handle None frame_selector
        if frame_selector is None:
            frame_selector = "全部"

        # Build a GLB file name
        glbfile = os.path.join(
            target_dir,
            f"glbscene_{frame_selector.replace('.', '_').replace(':', '').replace(' ', '_')}_cam{show_camera}_mesh{show_mesh}.glb",
        )

        # Convert predictions to GLB
        glbscene = convert_predictions_to_glb_scene(
            predictions,
            filter_by_frames=frame_selector,
            show_camera=show_camera,
            mask_sky_bg=filter_sky_bg,
            as_mesh=show_mesh,  # Use the show_mesh parameter
            mask_ambiguous=filter_ambiguous
        )
        glbscene.export(file_obj=glbfile)
        
        end_time = time.time()
        print(f"总时间: {end_time - start_time:.2f} 秒")
        log_msg = (
            f"重建成功 ({len(all_files)} 帧)。等待可视化。"
        )
        # Convert predictions to 3dgs ply
        gs_file = None
        splat_mode = 'ply'
        if "splats" in predictions:
            # Get Gaussian parameters (already filtered by GaussianSplatRenderer)
            means = predictions["splats"]["means"][0].reshape(-1, 3)
            scales = predictions["splats"]["scales"][0].reshape(-1, 3)
            quats = predictions["splats"]["quats"][0].reshape(-1, 4)
            colors = (predictions["splats"]["sh"][0] if "sh" in predictions["splats"] else predictions["splats"]["colors"][0]).reshape(-1, 3)
            opacities = predictions["splats"]["opacities"][0].reshape(-1)
            
            # Convert to torch tensors if needed
            if not isinstance(means, torch.Tensor):
                means = torch.from_numpy(means)
            if not isinstance(scales, torch.Tensor):
                scales = torch.from_numpy(scales)
            if not isinstance(quats, torch.Tensor):
                quats = torch.from_numpy(quats)
            if not isinstance(colors, torch.Tensor):
                colors = torch.from_numpy(colors)
            if not isinstance(opacities, torch.Tensor):
                opacities = torch.from_numpy(opacities)
            
            if splat_mode == 'ply':
                gs_file = os.path.join(target_dir, "gaussians.ply")
                save_gs_ply(
                    gs_file,
                    means,
                    scales,
                    quats,
                    colors,
                    opacities
                )
                print(f"高斯泼溅 PLY 已保存到: {gs_file}")
                print(f"文件存在: {os.path.exists(gs_file)}")
                if os.path.exists(gs_file):
                    print(f"文件大小: {os.path.getsize(gs_file)} 字节")
            elif splat_mode == 'splat':
                # Save Gaussian splat
                plydata = convert_gs_to_ply(
                        means,
                        scales,
                        quats,
                        colors,
                        opacities
                    )
                gs_file = os.path.join(target_dir, "gaussians.splat")
                gs_file = process_ply_to_splat(plydata, gs_file)

        # Initialize depth and normal view displays with processed data
        depth_vis, normal_vis = initialize_depth_normal_views(
            processed_data
        )

        # Update view selectors and info displays based on available views
        depth_slider, normal_slider, depth_info, normal_info = update_view_selectors(
            processed_data
        )

        # Automatically generate render video
        # Generate render video if possible
        rgb_video_path = None
        depth_video_path = None
        
        if "splats" in predictions:
            # try:
            from pathlib import Path
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Get camera parameters and image dimensions
            camera_poses = torch.tensor(predictions['camera_poses']).unsqueeze(0).to(device)
            camera_intrs = torch.tensor(predictions['camera_intrs']).unsqueeze(0).to(device)
            H, W = predictions['images'].shape[1], predictions['images'].shape[2]
            
            # Render video
            out_path = Path(target_dir) / "rendered_video"
            render_interpolated_video(
                model.gs_renderer, 
                predictions["splats"], 
                camera_poses, 
                camera_intrs, 
                (H, W), 
                out_path, 
                interp_per_pair=15, 
                loop_reverse=True,
                save_mode="split"
            )
            
            # Check output files
            rgb_video_path = str(out_path) + "_rgb.mp4"
            depth_video_path = str(out_path) + "_depth.mp4"
            
            if not os.path.exists(rgb_video_path) and not os.path.exists(depth_video_path):
                rgb_video_path = None
                depth_video_path = None
                
        # Cleanup
        del predictions
        gc.collect()
        torch.cuda.empty_cache()

        # Get terminal output and restore stdout
        terminal_log = tee.getvalue()
        sys.stdout = old_stdout

        # 检查生成的文件并设置下载按钮状态
        depth_dir = os.path.join(target_dir, "depth")
        normal_dir = os.path.join(target_dir, "normal")
        points_file = os.path.join(target_dir, "filtered_points.ply")
        colmap_dir = os.path.join(target_dir, "sparse", "0")
        gaussians_file = os.path.join(target_dir, "gaussians.ply")
        
        # 创建下载文件路径（如果存在）
        depth_zip = os.path.join(target_dir, "depth_maps.zip") if os.path.exists(depth_dir) else None
        normal_zip = os.path.join(target_dir, "normal_maps.zip") if os.path.exists(normal_dir) else None
        colmap_zip = os.path.join(target_dir, "colmap_reconstruction.zip") if os.path.exists(colmap_dir) else None
        
        # 如果文件存在，创建压缩包
        if depth_zip and os.path.exists(depth_dir):
            import zipfile
            with zipfile.ZipFile(depth_zip, 'w') as zipf:
                for root, dirs, files in os.walk(depth_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file), file)
        
        if normal_zip and os.path.exists(normal_dir):
            import zipfile
            with zipfile.ZipFile(normal_zip, 'w') as zipf:
                for root, dirs, files in os.walk(normal_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file), file)
        
        if colmap_zip and os.path.exists(colmap_dir):
            import zipfile
            with zipfile.ZipFile(colmap_zip, 'w') as zipf:
                for root, dirs, files in os.walk(colmap_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file), file)

        return (
            glbfile,
            log_msg,
            gr.Dropdown(choices=frame_selector_choices, value=frame_selector, interactive=True),
            processed_data,
            depth_vis,
            normal_vis,
            depth_slider,
            normal_slider,
            depth_info,
            normal_info,
            camera_params_file,
            gs_file,
            rgb_video_path,
            depth_video_path,
            terminal_log,
            gr.update(value=depth_zip, visible=depth_zip is not None),  # depth_download
            gr.update(value=normal_zip, visible=normal_zip is not None),  # normal_download
            gr.update(value=points_file, visible=os.path.exists(points_file)),  # points_download
            gr.update(value=colmap_zip, visible=colmap_zip is not None),  # colmap_download
            gr.update(value=gaussians_file, visible=os.path.exists(gaussians_file)),  # gaussians_download
        )
    
    except Exception as e:
        # In case of error, still restore stdout
        terminal_log = tee.getvalue()
        sys.stdout = old_stdout
        print(f"发生错误: {e}")
        raise


# -------------------------------------------------------------------------
# Helper functions for visualization
# -------------------------------------------------------------------------
def render_depth_visualization(depth_map, mask=None):
    """生成带有遮罩功能的彩色编码深度可视化图像"""
    if depth_map is None:
        return None

    # Create working copy and identify positive depth values
    depth_copy = depth_map.copy()
    positive_depth_mask = depth_copy > 0

    # Combine with user-provided mask for filtering
    if mask is not None:
        positive_depth_mask = positive_depth_mask & mask

    # Perform percentile-based normalization on valid regions
    if positive_depth_mask.sum() > 0:
        valid_depth_values = depth_copy[positive_depth_mask]
        lower_bound = np.percentile(valid_depth_values, 5)
        upper_bound = np.percentile(valid_depth_values, 95)

        depth_copy[positive_depth_mask] = (depth_copy[positive_depth_mask] - lower_bound) / (upper_bound - lower_bound)

    # Convert to RGB using matplotlib colormap
    import matplotlib.pyplot as plt

    color_mapper = plt.cm.turbo_r
    rgb_result = color_mapper(depth_copy)
    rgb_result = (rgb_result[:, :, :3] * 255).astype(np.uint8)

    # Mark invalid regions with white color
    rgb_result[~positive_depth_mask] = [255, 255, 255]

    return rgb_result

def render_normal_visualization(normal_map, mask=None):
    """将表面法向量转换为RGB颜色表示以供显示"""
    if normal_map is None:
        return None

    # Make a working copy to avoid modifying original data
    normal_display = normal_map.copy()

    # Handle masking by zeroing out invalid regions
    if mask is not None:
        masked_regions = ~mask
        normal_display[masked_regions] = [0, 0, 0]  # Zero out masked pixels

    # Transform from [-1, 1] to [0, 1] range for RGB display
    normal_display = (normal_display + 1.0) / 2.0
    normal_display = (normal_display * 255).astype(np.uint8)

    return normal_display


def clear_fields():
    """
    清除3D查看器、存储的target_dir并清空图库。
    """
    return None


def update_log():
    """
    在等待时显示快速日志消息。
    """
    return ""


def get_terminal_output():
    """
    获取当前终端输出以供实时显示
    """
    global current_terminal_output
    return current_terminal_output

# -------------------------------------------------------------------------
# FunctionExample scene metadata extraction
# -------------------------------------------------------------------------
def extract_example_scenes_metadata(base_directory):
    """
    提取包含有效图像的所有场景目录的综合元数据。
    
    Args:
        base_directory: 示例场景目录所在的根路径
        
    Returns:
        包含场景详情的字典集合（标题、位置、预览等）
    """
    from glob import glob
    
    # Return empty list if base directory is missing
    if not os.path.exists(base_directory):
        return []
    
    # Define supported image format extensions
    VALID_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
    
    scenes_data = []
    
    # Process each subdirectory in the base directory
    for directory_name in sorted(os.listdir(base_directory)):
        current_directory = os.path.join(base_directory, directory_name)
        
        # Filter out non-directory items
        if not os.path.isdir(current_directory):
            continue
        
        # Gather all valid image files within the current directory
        discovered_images = []
        for file_format in VALID_IMAGE_FORMATS:
            # Include both lowercase and uppercase format variations
            discovered_images.extend(glob(os.path.join(current_directory, f'*.{file_format}')))
            discovered_images.extend(glob(os.path.join(current_directory, f'*.{file_format.upper()}')))
        
        # Skip directories without any valid images
        if not discovered_images:
            continue
        
        # Ensure consistent image ordering
        discovered_images.sort()
        
        # Construct scene metadata record
        scene_record = {
            'name': directory_name,
            'path': current_directory,
            'thumbnail': discovered_images[0],
            'num_images': len(discovered_images),
            'image_files': discovered_images,
        }
        
        scenes_data.append(scene_record)
    
    return scenes_data

def load_example_scenes(scene_name, scenes):
    """
    初始化并准备示例场景进行3D重建处理。
    
    Args:
        scene_name: 要加载的目标场景标识符
        scenes: 包含所有可用场景配置的列表
        
    Returns:
        包含处理后的场景数据和状态信息的元组
    """
    # Locate the target scene configuration by matching names
    target_scene_config = None
    for scene_config in scenes:
        if scene_config["name"] == scene_name:
            target_scene_config = scene_config
            break

    # Handle case where requested scene doesn't exist
    if target_scene_config is None:
        return None, None, None, "场景未找到"

    # Prepare image file paths for processing pipeline
    # Extract all image file paths from the selected scene
    image_file_paths = []
    for img_file_path in target_scene_config["image_files"]:
        image_file_paths.append(img_file_path)

    # Process the scene images through the standard upload pipeline
    processed_target_dir, processed_image_list = process_uploaded_files(image_file_paths, 1.0)

    # Return structured response with scene data and user feedback
    status_message = f"成功加载场景 '{scene_name}'，包含 {target_scene_config['num_images']} 张图像。点击'重建'开始3D处理。"
    
    return (
        None,  # Reset reconstruction visualization
        None,  # Reset gaussian splatting output
        processed_target_dir,  # Provide working directory path
        processed_image_list,  # Update image gallery display
        status_message,
    )


# -------------------------------------------------------------------------
# UI and event handling
# -------------------------------------------------------------------------
theme = gr.themes.Base()

with gr.Blocks(
    title="网页端3DGS重建工具",
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #a9b8f8 0%, #7081e8 60%, #4254c5 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    .normal-weight-btn button,
    .normal-weight-btn button span,
    .normal-weight-btn button *,
    .normal-weight-btn * {
        font-weight: 400 !important;
    }
    .terminal-output {
        max-height: 400px !important;
        overflow-y: auto !important;
    }
    .terminal-output textarea {
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
        color: #333 !important;
        background-color: #f8f9fa !important;
        max-height: 400px !important;
    }
    .example-gallery {
        width: 100% !important;
    }
    .example-gallery img {
        width: 100% !important;
        height: 280px !important;
        object-fit: contain !important;
        aspect-ratio: 16 / 9 !important;
    }
    .example-gallery .grid-wrap {
        width: 100% !important;
    }
    
    /* 滑块导航样式 */
    .depth-tab-improved .gradio-slider input[type="range"] {
        height: 8px !important;
        border-radius: 4px !important;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }

    .depth-tab-improved .gradio-slider input[type="range"]::-webkit-slider-thumb {
        height: 20px !important;
        width: 20px !important;
        border-radius: 50% !important;
        background: #fff !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
    }

    .depth-tab-improved button {
        transition: all 0.3s ease !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }

    .depth-tab-improved button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    .normal-tab-improved .gradio-slider input[type="range"] {
        height: 8px !important;
        border-radius: 4px !important;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }

    .normal-tab-improved .gradio-slider input[type="range"]::-webkit-slider-thumb {
        height: 20px !important;
        width: 20px !important;
        border-radius: 50% !important;
        background: #fff !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
    }

    .normal-tab-improved button {
        transition: all 0.3s ease !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }

    .normal-tab-improved button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }

    #depth-view-info, #normal-view-info {
        animation: fadeIn 0.5s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    """
) as demo:
    # State variables for the tabbed interface
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")
    processed_data_state = gr.State(value=None)
    current_view_index = gr.State(value=0)  # Track current view index for navigation

    output_path_state = gr.Textbox(label="输出路径", visible=False, value="None")

    # Main UI components - 重新调整布局
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.Markdown("### 文件上传区域")
            file_upload = gr.File(
                file_count="multiple",
                label="上传视频或图片",
                interactive=True,
                file_types=["image", "video"],
                height="200px",
            )
            time_interval = gr.Slider(
                minimum=0.1,
                maximum=10.0,
                value=1.0,
                step=0.1,
                label="视频采样间隔（秒）",
                interactive=True,
                visible=True,
                scale=4,
            )
            resample_btn = gr.Button(
                "重新采样",
                visible=True,
                scale=1,
                elem_classes=["normal-weight-btn"],
            )
            image_gallery = gr.Gallery(
                label="图片预览",
                columns=4,
                height="200px",
                show_download_button=True,
                object_fit="contain",
                preview=True
            )
            
            terminal_output = gr.Textbox(
                label="终端输出",
                lines=6,
                max_lines=6,
                interactive=False,
                show_copy_button=True,
                container=True,
                elem_classes=["terminal-output"],
                autoscroll=True
            )

        with gr.Column(scale=4):
            log_output = gr.Markdown(
                "请先上传视频或图片，然后点击重建开始处理",
                elem_classes=["custom-log"],
            )

            with gr.Tabs() as tabs:
                with gr.Tab("3D高斯泼溅", id=1) as gs_tab:
                    with gr.Row():
                        with gr.Column(scale=3):
                            gs_output = gr.Model3D(
                                label="高斯泼溅成像",
                                height=500,
                            )
                        with gr.Column(scale=1):
                            gs_rgb_video = gr.Video(
                                label="渲染的RGB视频",
                                height=250,
                                autoplay=False,
                                loop=False,
                                interactive=False,
                            )
                            gs_depth_video = gr.Video(
                                label="渲染的深度视频",
                                height=250,
                                autoplay=False,
                                loop=False,
                                interactive=False,
                            )
                with gr.Tab("点云/网格", id=0):
                    reconstruction_output = gr.Model3D(
                        label="3D点云/网格",
                        height=500,
                        zoom_speed=0.4,
                        pan_speed=0.4,
                    )
                with gr.Tab("深度图", elem_classes=["depth-tab-improved"]):
                    depth_view_info = gr.HTML(
                        value="<div style='text-align: center; padding: 10px; background: #f8f8f8; color: #999; border-radius: 8px; margin-bottom: 10px;'>"
                              "<strong>深度视图导航</strong> | 当前: 视图 1 / 1 视图</div>",
                        elem_id="depth-view-info"
                    )
                    depth_view_slider = gr.Slider(
                        minimum=1, 
                        maximum=1, 
                        step=1, 
                        value=1,
                        label="视图选择滑块",
                        interactive=True,
                        elem_id="depth-view-slider"
                    )
                    depth_map = gr.Image(
                        type="numpy",
                        label="深度图",
                        format="png",
                        interactive=False,
                        height=340
                    )
                with gr.Tab("法线图", elem_classes=["normal-tab-improved"]):
                    normal_view_info = gr.HTML(
                        value="<div style='text-align: center; padding: 10px; background: #f8f8f8; color: #999; border-radius: 8px; margin-bottom: 10px;'>"
                              "<strong>法线视图导航</strong> | 当前: 视图 1 / 1 视图</div>",
                        elem_id="normal-view-info"
                    )
                    normal_view_slider = gr.Slider(
                        minimum=1, 
                        maximum=1, 
                        step=1, 
                        value=1,
                        label="视图选择滑块",
                        interactive=True,
                        elem_id="normal-view-slider"
                    )
                    normal_map = gr.Image(
                        type="numpy",
                        label="法线图",
                        format="png",
                        interactive=False,
                        height=340
                    )
                with gr.Tab("相机参数", elem_classes=["camera-tab"]):
                    with gr.Row():
                        gr.HTML("")
                        camera_params = gr.DownloadButton(
                            label="下载相机参数",
                            scale=1,
                            variant="primary",
                        )
                        gr.HTML("")
                
                with gr.Tab("高级输出"):
                    gr.Markdown("### 高级功能生成的文件")
                    with gr.Row():
                        depth_download = gr.DownloadButton(
                            label="下载深度图",
                            scale=1,
                            visible=False,
                        )
                        normal_download = gr.DownloadButton(
                            label="下载法线图", 
                            scale=1,
                            visible=False,
                        )
                        points_download = gr.DownloadButton(
                            label="下载点云PLY",
                            scale=1,
                            visible=False,
                        )
                    with gr.Row():
                        colmap_download = gr.DownloadButton(
                            label="下载COLMAP格式",
                            scale=1,
                            visible=False,
                        )
                        gaussians_download = gr.DownloadButton(
                            label="下载3D高斯PLY",
                            scale=1,
                            visible=False,
                        )
                    
            with gr.Row():
                reconstruct_btn = gr.Button(
                    "重建", 
                    scale=1, 
                    variant="primary"
                )
                clear_btn = gr.ClearButton(
                    [
                        file_upload,
                        reconstruction_output,
                        log_output,
                        output_path_state,
                        image_gallery,
                        depth_map,
                        normal_map,
                        depth_view_slider,
                        normal_view_slider,
                        depth_view_info,
                        normal_view_info,
                        camera_params,
                        gs_output,
                        gs_rgb_video,
                        gs_depth_video,
                        depth_download,
                        normal_download,
                        points_download,
                        colmap_download,
                        gaussians_download,
                    ],
                    scale=1,
                )
                
            with gr.Row():
                frame_selector = gr.Dropdown(
                        choices=["全部"], value="全部", label="显示特定帧的点"
                    )
                
            gr.Markdown("### 重建选项: (不应用于3DGS)")
            with gr.Row():
                show_camera = gr.Checkbox(label="显示相机", value=True)
                show_mesh = gr.Checkbox(label="显示网格", value=True)
                filter_ambiguous = gr.Checkbox(label="过滤低置信度 & 深度/法线边缘", value=True)
                filter_sky_bg = gr.Checkbox(label="过滤天空背景", value=False)

            gr.Markdown("### 高级功能选项")
            with gr.Row():
                conf_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=100.0,
                    value=0.0,
                    step=1.0,
                    label="置信度阈值 (%)",
                    info="过滤低置信度点云，0表示不过滤"
                )
                target_size = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=518,
                    step=32,
                    label="目标图像尺寸",
                    info="调整图像处理分辨率"
                )
            
            with gr.Row():
                save_depth_maps = gr.Checkbox(label="保存深度图", value=True)
                save_normal_maps = gr.Checkbox(label="保存法线图", value=True)
                save_point_cloud = gr.Checkbox(label="保存点云PLY", value=True)
                save_gaussians = gr.Checkbox(label="保存3D高斯PLY", value=True)
            
            with gr.Row():
                save_colmap = gr.Checkbox(label="导出COLMAP格式", value=False)
                save_rendered_video = gr.Checkbox(label="生成渲染视频", value=True)
                enable_amp = gr.Checkbox(label="启用混合精度", value=True)
            
            with gr.Row():
                cond_pose = gr.Checkbox(label="使用相机位姿先验", value=False)
                cond_intrinsics = gr.Checkbox(label="使用内参先验", value=False)
                cond_depth = gr.Checkbox(label="使用深度先验", value=False)

        with gr.Column(scale=1):            
            gr.Markdown("### 点击加载示例场景")
            realworld_scenes = extract_example_scenes_metadata("examples/realistic") if os.path.exists("examples/realistic") else extract_example_scenes_metadata("examples")
            generated_scenes = extract_example_scenes_metadata("examples/stylistic") if os.path.exists("examples/stylistic") else []
            
            # If no subdirectories exist, fall back to single gallery
            if not os.path.exists("examples/realistic") and not os.path.exists("examples/stylistic"):
                # Fallback: use all scenes from examples directory
                all_scenes = extract_example_scenes_metadata("examples")
                if all_scenes:
                    gallery_items = [
                        (scene["thumbnail"], f"{scene['name']}\n{scene['num_images']} 张图像")
                        for scene in all_scenes
                    ]
                    
                    example_gallery = gr.Gallery(
                        value=gallery_items,
                        label="示例场景",
                        columns=1,
                        rows=None,
                        height=800,
                        object_fit="contain",
                        show_label=False,
                        interactive=True,
                        preview=False,
                        allow_preview=False,
                        elem_classes=["example-gallery"]
                    )
                    
                    def handle_example_selection(evt: gr.SelectData):
                        if evt:
                            result = load_example_scenes(all_scenes[evt.index]["name"], all_scenes)
                            return result
                        return (None, None, None, None, "未选择场景")
                    
                    example_gallery.select(
                        fn=handle_example_selection,
                        outputs=[
                            reconstruction_output,
                            gs_output,
                            output_path_state,
                            image_gallery,
                            log_output,
                        ],
                    )
            else:
                # Tabbed interface for categorized examples
                with gr.Tabs():
                    with gr.Tab("真实场景"):
                        if realworld_scenes:
                            realworld_items = [
                                (scene["thumbnail"], f"{scene['name']}\n {scene['num_images']} 张图像")
                                for scene in realworld_scenes
                            ]
                            
                            realworld_gallery = gr.Gallery(
                                value=realworld_items,
                                label="真实场景示例",
                                columns=1,
                                rows=None,
                                height=750,
                                object_fit="contain",
                                show_label=False,
                                interactive=True,
                                preview=False,
                                allow_preview=False,
                                elem_classes=["example-gallery"]
                            )
                            
                            def handle_realworld_selection(evt: gr.SelectData):
                                if evt:
                                    result = load_example_scenes(realworld_scenes[evt.index]["name"], realworld_scenes)
                                    return result
                                return (None, None, None, None, "未选择场景")
                            
                            realworld_gallery.select(
                                fn=handle_realworld_selection,
                                outputs=[
                                    reconstruction_output,
                                    gs_output,
                                    output_path_state,
                                    image_gallery,
                                    log_output,
                                ],
                            )
                        else:
                            gr.Markdown("暂无真实世界示例")
                    
                    with gr.Tab("风格化场景"):
                        if generated_scenes:
                            generated_items = [
                                (scene["thumbnail"], f"{scene['name']}\n📷 {scene['num_images']} 张图像")
                                for scene in generated_scenes
                            ]
                            
                            generated_gallery = gr.Gallery(
                                value=generated_items,
                                label="风格化场景示例",
                                columns=1,
                                rows=None,
                                height=750,
                                object_fit="contain",
                                show_label=False,
                                interactive=True,
                                preview=False,
                                allow_preview=False,
                                elem_classes=["example-gallery"]
                            )
                            
                            def handle_generated_selection(evt: gr.SelectData):
                                if evt:
                                    result = load_example_scenes(generated_scenes[evt.index]["name"], generated_scenes)
                                    return result
                                return (None, None, None, None, "未选择场景")
                            
                            generated_gallery.select(
                                fn=handle_generated_selection,
                                outputs=[
                                    reconstruction_output,
                                    gs_output,
                                    output_path_state,
                                    image_gallery,
                                    log_output,
                                ],
                            )
                        else:
                            gr.Markdown("暂无生成示例")
    
    # -------------------------------------------------------------------------
    # Click logic
    # -------------------------------------------------------------------------
    reconstruct_btn.click(fn=clear_fields, inputs=[], outputs=[]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[
            output_path_state,
            frame_selector,
            show_camera,
            filter_sky_bg,
            show_mesh,
            filter_ambiguous,
            conf_threshold,
            target_size,
            save_depth_maps,
            save_normal_maps,
            save_point_cloud,
            save_gaussians,
            save_colmap,
            save_rendered_video,
            enable_amp,
            cond_pose,
            cond_intrinsics,
            cond_depth,
        ],
        outputs=[
            reconstruction_output,
            log_output,
            frame_selector,
            processed_data_state,
            depth_map,
            normal_map,
            depth_view_slider,
            normal_view_slider,
            depth_view_info,
            normal_view_info,
            camera_params,
            gs_output,
            gs_rgb_video,
            gs_depth_video,
            terminal_output,
            depth_download,
            normal_download,
            points_download,
            colmap_download,
            gaussians_download,
        ],
    ).then(
        fn=lambda: "False",
        inputs=[],
        outputs=[is_example],  # set is_example to "False"
    )

    # -------------------------------------------------------------------------
    # Live update logic
    # -------------------------------------------------------------------------
    def refresh_3d_scene(
        workspace_path,
        frame_selector,
        show_camera,
        is_example,
        filter_sky_bg=False,
        show_mesh=False,
        filter_ambiguous=False
    ):
        """
        刷新3D场景可视化
        
        从工作区加载预测数据，根据当前参数生成或重用GLB场景文件，
        并返回3D查看器所需的文件路径。
        
        Args:
            workspace_path: 重建结果的工作区目录路径
            frame_selector: 用于从特定帧过滤点的帧选择器值
            show_camera: 是否显示相机位置
            is_example: 这是否是示例场景
            filter_sky_bg: 是否过滤天空背景
            show_mesh: 是否以网格模式显示
            filter_ambiguous: 是否过滤低置信度模糊区域
            
        Returns:
            tuple: (GLB场景文件路径, 高斯点云文件路径, 状态消息)
        """

        # If example scene is clicked, skip processing directly
        if is_example == "True":
            return (
                gr.update(),
                gr.update(),
                "暂无重建结果。请先点击重建按钮。",
            )

        # Validate workspace directory path
        if not workspace_path or workspace_path == "None" or not os.path.isdir(workspace_path):
            return (
                gr.update(),
                gr.update(),
                "暂无重建结果。请先点击重建按钮。",
            )

        # Check if prediction data file exists
        prediction_file_path = os.path.join(workspace_path, "predictions.npz")
        if not os.path.exists(prediction_file_path):
            return (
                gr.update(),
                gr.update(),
                f"预测文件不存在: {prediction_file_path}。请先运行重建。",
            )

        # Load prediction data
        prediction_data = np.load(prediction_file_path, allow_pickle=True)
        predictions = {key: prediction_data[key] for key in prediction_data.keys() if key != 'splats'}

        # Generate GLB scene file path (named based on parameter combination)
        safe_frame_name = frame_selector.replace('.', '_').replace(':', '').replace(' ', '_')
        scene_filename = f"scene_{safe_frame_name}_cam{show_camera}_mesh{show_mesh}_edges{filter_ambiguous}_sky{filter_sky_bg}.glb"
        scene_glb_path = os.path.join(workspace_path, scene_filename)

        # If GLB file doesn't exist, generate new scene file
        if not os.path.exists(scene_glb_path):
            scene_model = convert_predictions_to_glb_scene(
                predictions,
                filter_by_frames=frame_selector,
                show_camera=show_camera,
                mask_sky_bg=filter_sky_bg,
                as_mesh=show_mesh,
                mask_ambiguous=filter_ambiguous
            )
            scene_model.export(file_obj=scene_glb_path)

        # Find Gaussian point cloud file
        gaussian_file_path = os.path.join(workspace_path, "gaussians.ply")
        if not os.path.exists(gaussian_file_path):
            gaussian_file_path = None

        return (
            scene_glb_path,
            gaussian_file_path,
            "3D场景已更新。",
        )
    
    def refresh_view_displays_on_filter_update(
        workspace_dir,
        sky_background_filter,
        current_processed_data,
        depth_slider_position,
        normal_slider_position,
    ):
        """
        当过滤器设置更改时刷新深度和法线视图显示
        
        当背景过滤器复选框状态更改时，重新生成处理数据并更新所有视图显示。
        这确保过滤器效果在深度图和法线图可视化中实时反映。
        
        Args:
            workspace_dir: 包含预测数据和图像的工作区目录路径
            sky_background_filter: 天空背景过滤器启用状态
            current_processed_data: 当前处理的可视化数据
            depth_slider_position: 深度视图滑块的当前位置
            normal_slider_position: 法线视图滑块的当前位置
            
        Returns:
            tuple: (更新的处理数据, 深度可视化结果, 法线可视化结果)
        """
        
        # Validate workspace directory validity
        if not workspace_dir or workspace_dir == "None" or not os.path.isdir(workspace_dir):
            return current_processed_data, None, None

        # Build and check prediction data file path
        prediction_data_path = os.path.join(workspace_dir, "predictions.npz")
        if not os.path.exists(prediction_data_path):
            return current_processed_data, None, None

        try:
            # Load raw prediction data
            raw_prediction_data = np.load(prediction_data_path, allow_pickle=True)
            predictions_dict = {key: raw_prediction_data[key] for key in raw_prediction_data.keys()}

            # Load image data using WorldMirror's load_images function
            images_directory = os.path.join(workspace_dir, "images")
            image_file_paths = [os.path.join(images_directory, path) for path in os.listdir(images_directory)]
            img = load_and_preprocess_images(image_file_paths)
            img = img.detach().cpu().numpy()

            # Regenerate processed data with new filter settings
            refreshed_data = {}
            for view_idx in range(img.shape[1]):
                view_data = {
                    "image": img[0, view_idx],
                    "points3d": predictions_dict["world_points"][view_idx],
                    "depth": None,
                    "normal": None,
                    "mask": None,
                }
                mask = predictions_dict["final_mask"][view_idx].copy()
                if sky_background_filter:
                    sky_mask = predictions_dict["sky_mask"][view_idx]
                    mask = mask & sky_mask
                view_data["mask"] = mask
                view_data["depth"] = predictions_dict["depth"][view_idx].squeeze()
                view_data["normal"] = predictions_dict["normal"][view_idx]
                refreshed_data[view_idx] = view_data

            # Get current view indices from slider positions (convert to 0-based indices)
            current_depth_index = int(depth_slider_position) - 1 if depth_slider_position else 0
            current_normal_index = int(normal_slider_position) - 1 if normal_slider_position else 0

            # Update depth and normal views with new filter data
            updated_depth_visualization = update_depth_view(refreshed_data, current_depth_index)
            updated_normal_visualization = update_normal_view(refreshed_data, current_normal_index)

            return refreshed_data, updated_depth_visualization, updated_normal_visualization

        except Exception as error:
            print(f"刷新视图显示时发生错误: {error}")
            return current_processed_data, None, None

    frame_selector.change(
        refresh_3d_scene,
        [
            output_path_state,
            frame_selector,
            show_camera,
            is_example,
            filter_sky_bg,
            show_mesh,
            filter_ambiguous
        ],
        [reconstruction_output, gs_output, log_output],
    )
    show_camera.change(
        refresh_3d_scene,
        [
            output_path_state,
            frame_selector,
            show_camera,
            is_example,
            filter_sky_bg,
            show_mesh,
            filter_ambiguous
        ],
        [reconstruction_output, gs_output, log_output],
    )
    show_mesh.change(
        refresh_3d_scene,
        [
            output_path_state,
            frame_selector,
            show_camera,
            is_example,
            filter_sky_bg,
            show_mesh,
            filter_ambiguous
        ],
        [reconstruction_output, gs_output, log_output],
    )
    
    filter_sky_bg.change(
        refresh_3d_scene,
        [
            output_path_state,
            frame_selector,
            show_camera,
            is_example,
            filter_sky_bg,
            show_mesh,
            filter_ambiguous
        ],
        [reconstruction_output, gs_output, log_output],
    ).then(
        fn=refresh_view_displays_on_filter_update,
        inputs=[
            output_path_state,
            filter_sky_bg,
            processed_data_state,
            depth_view_slider,
            normal_view_slider,
        ],
        outputs=[
            processed_data_state,
            depth_map,
            normal_map,
        ],
    )
    filter_ambiguous.change(
        refresh_3d_scene,
        [
            output_path_state,
            frame_selector,
            show_camera,
            is_example,
            filter_sky_bg,
            show_mesh,
            filter_ambiguous
        ],
        [reconstruction_output, gs_output, log_output],
    ).then(
        fn=refresh_view_displays_on_filter_update,
        inputs=[
            output_path_state,
            filter_sky_bg,
            processed_data_state,
            depth_view_slider,
            normal_view_slider,
        ],
        outputs=[
            processed_data_state,
            depth_map,
            normal_map,
        ],
    )

    # -------------------------------------------------------------------------
    # Auto update gallery when user uploads or changes files
    # -------------------------------------------------------------------------
    def update_gallery_on_file_upload(files, interval):
        if not files:
            return None, None, None, ""
        
        # Capture terminal output
        tee = TeeOutput()
        old_stdout = sys.stdout
        sys.stdout = tee
        
        try:
            target_dir, image_paths = process_uploaded_files(files, interval)
            terminal_log = tee.getvalue()
            sys.stdout = old_stdout
            
            return (
                target_dir,
                image_paths,
                "上传完成。点击'重建'开始3D处理。",
                terminal_log,
            )
        except Exception as e:
            terminal_log = tee.getvalue()
            sys.stdout = old_stdout
            print(f"发生错误: {e}")
            raise

    def resample_video_with_new_interval(files, new_interval, current_target_dir):
        """使用新的滑块值重新采样视频"""
        if not files:
            return (
                current_target_dir,
                None,
                "没有文件需要重新采样。",
                "",
            )

        # Check if we have videos to resample
        video_extensions = [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".wmv",
            ".flv",
            ".webm",
            ".m4v",
            ".3gp",
        ]
        has_video = any(
            os.path.splitext(
                str(file_data["name"] if isinstance(file_data, dict) else file_data)
            )[1].lower()
            in video_extensions
            for file_data in files
        )

        if not has_video:
            return (
                current_target_dir,
                None,
                "未找到需要重新采样的视频。",
                "",
            )

        # Capture terminal output
        tee = TeeOutput()
        old_stdout = sys.stdout
        sys.stdout = tee
        
        try:
            # Clean up old target directory if it exists
            if (
                current_target_dir
                and current_target_dir != "None"
                and os.path.exists(current_target_dir)
            ):
                shutil.rmtree(current_target_dir)

            # Process files with new interval
            target_dir, image_paths = process_uploaded_files(files, new_interval)
            
            terminal_log = tee.getvalue()
            sys.stdout = old_stdout

            return (
                target_dir,
                image_paths,
                f"视频已使用 {new_interval}秒间隔重新采样。点击'重建'开始3D处理。",
                terminal_log,
            )
        except Exception as e:
            terminal_log = tee.getvalue()
            sys.stdout = old_stdout
            print(f"发生错误: {e}")
            raise

    file_upload.change(
        fn=update_gallery_on_file_upload,
        inputs=[file_upload, time_interval],
        outputs=[output_path_state, image_gallery, log_output, terminal_output],
    )

    resample_btn.click(
        fn=resample_video_with_new_interval,
        inputs=[file_upload, time_interval, output_path_state],
        outputs=[output_path_state, image_gallery, log_output, terminal_output],
    )

    # -------------------------------------------------------------------------
    # Navigation for Depth, Normal tabs
    # -------------------------------------------------------------------------
    def navigate_with_slider(processed_data, target_view):
        """使用滑块导航到指定视图"""
        if processed_data is None or len(processed_data) == 0:
            return None, update_view_info(1, 1)
        
        # Check if target_view is None or invalid value, and safely convert to int
        try:
            if target_view is None:
                target_view = 1
            else:
                target_view = int(float(target_view))  # Convert to float first then int, handle decimal input
        except (ValueError, TypeError):
            target_view = 1
        
        total_views = len(processed_data)
        # Ensure view index is within valid range
        view_index = max(1, min(target_view, total_views)) - 1
        
        # Update depth map
        depth_vis = update_depth_view(processed_data, view_index)
        
        # Update view information
        info_html = update_view_info(view_index + 1, total_views)
        
        return depth_vis, info_html

    def navigate_with_slider_normal(processed_data, target_view):
        """使用滑块导航到指定法线视图"""
        if processed_data is None or len(processed_data) == 0:
            return None, update_view_info(1, 1, "法线")
        
        # Check if target_view is None or invalid value, and safely convert to int
        try:
            if target_view is None:
                target_view = 1
            else:
                target_view = int(float(target_view))  # Convert to float first then int, handle decimal input
        except (ValueError, TypeError):
            target_view = 1
        
        total_views = len(processed_data)
        # Ensure view index is within valid range
        view_index = max(1, min(target_view, total_views)) - 1
        
        # Update normal map
        normal_vis = update_normal_view(processed_data, view_index)
        
        # Update view information
        info_html = update_view_info(view_index + 1, total_views, "法线")
        
        return normal_vis, info_html

    def handle_depth_slider_change(processed_data, target_view):
        return navigate_with_slider(processed_data, target_view)
    
    def handle_normal_slider_change(processed_data, target_view):
        return navigate_with_slider_normal(processed_data, target_view)
    
    depth_view_slider.change(
        fn=handle_depth_slider_change,
        inputs=[processed_data_state, depth_view_slider],
        outputs=[depth_map, depth_view_info]
    )
    
    normal_view_slider.change(
        fn=handle_normal_slider_change,
        inputs=[processed_data_state, normal_view_slider],
        outputs=[normal_map, normal_view_info]
    )
    
    # -------------------------------------------------------------------------
    # Real-time terminal output update
    # -------------------------------------------------------------------------
    # Use a timer to periodically update terminal output
    timer = gr.Timer(value=0.5)  # Update every 0.5 seconds
    timer.tick(
        fn=get_terminal_output,
        inputs=[],
        outputs=[terminal_output]
    )
    

    demo.queue().launch(
        show_error=True,
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        favicon_path='assets/favicon.svg',
    )