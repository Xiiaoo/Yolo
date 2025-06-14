from ultralytics import YOLO
import torch, os, warnings,time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from loguru import logger
# 检查是否有可用的 GPU
# if torch.cuda.is_available():
#     # 获取 GPU 数量
#     num_gpus = torch.cuda.device_count()
#     print(f"Number of available GPUs: {num_gpus}")

#     # 遍历所有 GPU 并打印设备名
#     for i in range(num_gpus):
#         print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
# else:
#     print("No GPU available.")

# Load a model



if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
    warnings.filterwarnings("ignore", category=UserWarning)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # transform = transforms.Compose([
    #     transforms.Resize((320, 320)),
    #     transforms.ToTensor(),
    # ])
    # dataset = datasets.ImageFolder(root='D:\\Git\\Yolo\\yolo\\images', transform=transform)
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # gpu_data = []
    # for images, labels in dataloader:
    #     images = images.to(device)  # 将图片数据移动到 GPU
    #     labels = labels.to(device)  # 将标签数据移动到 GPU
    #     gpu_data.append((images, labels))

    model = YOLO("yolo11s-pose.yaml")
    model.load('yolo11s-pose.pt')
 
    # 定义训练参数，添加默认值、范围和中文注释
    train_params = {
        'data': "coco8-pose.yaml",   # 数据集配置文件路径，需要自定义修改
        'epochs': 200,               # 总训练轮次，默认值 100，范围 >= 1
        'imgsz': 320,               # 输入图像大小，默认值 640，范围 >= 32
        'batch': 16,                # 批次大小，默认值 16，范围 >= 1
        'save': True,               # 是否保存训练结果和模型，默认值 True
        'save_period': -1,          # 模型保存频率，默认值 -1，表示只保存最终结果
        'cache': False,             # 是否缓存数据集，默认值 False
        'device':device,             # 训练设备，默认值 None，支持 "cpu", "gpu"(device=0,1), "mps"
        'workers': 16,               # 数据加载线程数，默认值 8，影响数据预处理速度
        'project': "epfs/",            # 项目名称，保存训练结果的目录，默认值 None
        'name': "train",            # 训练运行的名称，用于创建子目录保存结果，默认值 None
        'optimizer': 'SGD',        # 优化器，默认值 'auto'，支持 'SGD', 'Adam', 'AdamW'
        'verbose': True,            # 是否启用详细日志输出，默认值 False
        'seed': 0,                  # 随机种子，确保结果的可重复性，默认值 0
        'deterministic': True,      # 是否强制使用确定性算法，默认值 True
        'single_cls': False,        # 是否将多类别数据集视为单一类别，默认值 False
        'rect': False,              # 是否启用矩形训练（优化批次图像大小），默认值 False
        'cos_lr': False,            # 是否使用余弦学习率调度器，默认值 False
        'close_mosaic': 10,         # 在最后 N 轮次中禁用 Mosaic 数据增强，默认值 10
        'resume': False,            # 是否从上次保存的检查点继续训练，默认值 False
        'amp': True,                # 是否启用自动混合精度（AMP）训练，默认值 True
        'fraction': 1.0,            # 使用数据集的比例，默认值 1.0
        'profile': False,           # 是否启用 ONNX 或 TensorRT 模型优化分析，默认值 False
        'freeze': None,             # 冻结模型的前 N 层，默认值 None
        'lr0': 0.01,                # 初始学习率，默认值 0.01，范围 >= 0
        'lrf': 0.01,                # 最终学习率与初始学习率的比值，默认值 0.01
        'momentum': 0.937,          # SGD 或 Adam 的动量因子，默认值 0.937，范围 [0, 1]
        'weight_decay': 0.0005,     # 权重衰减，防止过拟合，默认值 0.0005
        'warmup_epochs': 3.0,       # 预热学习率的轮次，默认值 3.0
        'warmup_momentum': 0.8,     # 预热阶段的初始动量，默认值 0.8
        'warmup_bias_lr': 0.1,      # 预热阶段的偏置学习率，默认值 0.1
        'box': 7.5,                 # 边框损失的权重，默认值 7.5
        'cls': 0.5,                 # 分类损失的权重，默认值 0.5
        'dfl': 1.5,                 # 分布焦点损失的权重，默认值 1.5
        'pose': 12.0,               # 姿态损失的权重，默认值 12.0
        'kobj': 1.0,                # 关键点目标损失的权重，默认值 1.0
        'label_smoothing': 0.0,     # 标签平滑处理，默认值 0.0
        'nbs': 64,                  # 归一化批次大小，默认值 64
        'overlap_mask': True,       # 是否在训练期间启用掩码重叠，默认值 True
        'mask_ratio': 4,            # 掩码下采样比例，默认值 4
        'dropout': 0.0,             # 随机失活率，用于防止过拟合，默认值 0.0
        'val': True,                # 是否在训练期间启用验证，默认值 True
        'plots': True,             # 是否生成训练曲线和验证指标图，默认值 True
        'patience': 30,            # 早停训练的耐心轮次，默认值 100
    
        # 数据增强相关参数
        # 'hsv_h': 0.2,             # 色相变化范围 (0.0 - 1.0)，默认值 0.015
        # 'hsv_s': 0.7,               # 饱和度变化范围 (0.0 - 1.0)，默认值 0.7
        # 'hsv_v': 0.4,               # 亮度变化范围 (0.0 - 1.0)，默认值 0.4
        # 'degrees': 30.0,             # 旋转角度范围 (-180 - 180)，默认值 0.0
        # 'translate': 0.1,           # 平移范围 (0.0 - 1.0)，默认值 0.1
        # 'scale': 0.5,               # 缩放比例范围 (>= 0.0)，默认值 0.5
        # 'shear': 0.0,               # 剪切角度范围 (-180 - 180)，默认值 0.0
        # 'perspective': 0.0,         # 透视变化范围 (0.0 - 0.001)，默认值 0.0
        # 'flipud': 0.0,              # 上下翻转概率 (0.0 - 1.0)，默认值 0.0
        # 'fliplr': 0.5,              # 左右翻转概率 (0.0 - 1.0)，默认值 0.5
        # 'bgr': 0.0,                 # BGR 色彩顺序调整概率 (0.0 - 1.0)，默认值 0.0
        # 'mosaic': 0.5,              # Mosaic 数据增强 (0.0 - 1.0)，默认值 1.0
        # 'mixup': 0.0,               # Mixup 数据增强 (0.0 - 1.0)，默认值 0.0
        # 'copy_paste': 0.0,          # Copy-Paste 数据增强 (0.0 - 1.0)，默认值 0.0
        # 'copy_paste_mode': 'flip',  # Copy-Paste 增强模式 ('flip' 或 'mixup')，默认值 'flip'
        # 'auto_augment': 'randaugment',  # 自动增强策略 ('randaugment', 'autoaugment', 'augmix')，默认值 'randaugment'
        # 'erasing': 0.4,             # 随机擦除增强比例 (0.0 - 0.9)，默认值 0.4
        # 'crop_fraction': 1.0,       # 裁剪比例 (0.1 - 1.0)，默认值 1.0
    
    }
    # 进行训练
    results = model.train(**train_params)
    # model.export(format='onnx', imgsz=320)
    
    
    # model = YOLO(
    #     "best.pt", 
    #     task='pose', 
    #     verbose=False
    # )
    
    # results = model(
    #     source="E:\\AI数据集\\京东\\images\\0c390e1cb1b3ce97c5adeccc563b5e5a.png",
    #     imgsz=320,
    #     show=True,
    #     show_boxes=True,
    #     device='cpu',
       
    # )
    # time.sleep(10)
    # logger.info(results[0].keypoints)
    # logger.info(results[0].boxes)
    
    # time.sleep(10)