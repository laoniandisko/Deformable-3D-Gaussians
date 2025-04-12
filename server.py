from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import json
import numpy as np
import torch
from gaussian_renderer import render
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state
from scene.cameras import Camera
from arguments import ModelParams, PipelineParams, get_combined_args
import cv2
import base64
from io import BytesIO
from PIL import Image
from argparse import ArgumentParser

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局变量存储场景
scene = None
camera = None
background = None
pipeline = None
deform = None
current_time = 0.0

# 创建 ArgumentParser 实例
parser = ArgumentParser()

def init_scene(model_path):
    global scene, camera, background, pipeline, deform
    # 初始化参数解析器
    parser = ArgumentParser()
    
    # 初始化模型参数
    model_params = ModelParams(parser)
    pipeline = PipelineParams(parser)
    
    # 解析参数
    args = parser.parse_args([])
    args.model_path = model_path
    args.source_path = model_path
    args.eval = True
    args.load2gpu_on_the_fly = False
    args.white_background = False
    args.data_device = "cuda"
    args.sh_degree = 3
    args.convert_SHs_python = False
    args.compute_cov3D_python = False
    args.debug = False
    args.source_path = "transforms_train.json"  # 设置为Blender数据集类型
    args.images = "images"
    args.resolution = -1
    args.white_background = True
    args.data_device = "cuda"
    
    # 设置背景颜色
    background = torch.zeros(3, device="cuda")
    
    # 初始化高斯模型
    gaussians = GaussianModel(3)
    # 加载模型参数
    gaussians.load_ply(model_path)
    
    # 初始化变形模型
    deform = DeformModel(gaussians)
    
    # 创建一个简单的场景，不依赖于数据集类型
    scene = type('Scene', (), {
        'gaussians': gaussians,
        'cameras_extent': 1.0
    })()
    
    # 创建空白图像
    blank_image = torch.zeros((3, 800, 800), device="cuda")
    
    # 初始化相机
    camera = Camera(
        colmap_id=0,
        R=np.eye(3),
        T=np.zeros(3),
        FoVx=0.6911112070083618,
        FoVy=0.6911112070083618,
        image=blank_image,
        gt_alpha_mask=None,
        image_name="web_camera",
        uid=0,
        data_device="cuda",
        fid=0
    )
    return scene, camera

@app.route('/')
def index():
    return render_template('index.html')

def generate_frame():
    global current_time
    while True:
        if scene is not None and camera is not None:
            # 获取高斯模型的当前位置
            xyz = scene.gaussians.get_xyz
            time_input = torch.tensor([[current_time]], device="cuda").expand(xyz.shape[0], 1)
            d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
            
            # 渲染当前帧
            render_pkg = render(camera, scene.gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, False)
            image = render_pkg["render"]
            
            # 转换为PIL图像
            image = image.detach().cpu().numpy()
            image = np.transpose(image, (1, 2, 0))  # 将通道维度移到最后
            image = Image.fromarray((image * 255).astype(np.uint8))
            
            # 转换为base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # 更新时间
            current_time = (current_time + 0.01) % 1.0
            
            # 发送图像数据
            yield f"data: {img_str}\n\n"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frame(),
                    mimetype='text/event-stream')

@socketio.on('update_camera')
def handle_camera_update(data):
    global camera, current_time
    if camera is not None:
        # 更新相机参数
        R = np.array(data['rotation'])
        T = np.array(data['translation'])
        camera.reset_extrinsic(R, T)
        
        # 更新时间
        if 'time' in data:
            current_time = float(data['time'])

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    # 初始化场景
    model_path = "output/jumpingjacks/point_cloud/iteration_40000/point_cloud.ply"  # 替换为实际的模型路径
    init_scene(model_path)
    
    # 启动服务器
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) 