import os
import argparse
import torch
import gc
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler,AutoencoderKL
from PIL import Image
from tqdm import tqdm
import numpy as np

# 设置命令行参数
parser = argparse.ArgumentParser("Stable Diffusion XL Image Generation")
parser.add_argument("--out", type=str, default="../Semanticprompt/dataset/ChestX/aug_base_XL", help="Output directory for generated images")
parser.add_argument("--base_model_path", type=str, default="SDXL_base", help="Path to the local SDXL base model directory")
parser.add_argument("--vae_model_path", type=str, default="SDXL_fix", help="Path to the local SDXL VAE fix model directory")
parser.add_argument("--txt_list", type=str, required=True, help="Path to the file containing category names")
parser.add_argument("--num-images-per-view", type=int, default=16, help="Number of images to generate per view")
parser.add_argument("--batch-size", type=int, default=1, help="Batch size for image generation")
parser.add_argument("--device", type=str, default="cuda:1", help="Device to use (e.g., cuda:0, cuda:1)")
parser.add_argument("--num-inference-steps", type=int, default=30, help="Number of inference steps")
parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale")


# 解析命令行参数
args = parser.parse_args()

# 创建输出目录
output_dir = args.out
os.makedirs(output_dir, exist_ok=True)

# 设置显存分配策略
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 打印环境变量
print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))

# 设置设备
device = torch.device(args.device)

# 显式设置当前设备
torch.cuda.set_device(device)

# 打印设备信息
print("Using device:", device)
print("CUDA device count:", torch.cuda.device_count())
print("Current CUDA device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(device))

# 获取显卡的总显存（以字节为单位）
total_memory = torch.cuda.get_device_properties(device).total_memory
total_memory_gb = total_memory / (1024 * 1024 * 1024)
print("Total Memory (GB):", total_memory_gb)

# 加载基础模型
print(f"Loading model from: {args.base_model_path}")


vae = AutoencoderKL.from_pretrained(args.vae_model_path,torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    args.base_model_path,
    torch_dtype=torch.float16,
    vae = vae,
    variant="fp16",
    use_safetensors=True
).to(device)
high_noise_frac = 0.7

print("Model loaded successfully")

# 使用 Euler a 采样器
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# 逐步添加优化设置
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.enable_attention_slicing(slice_size="auto")

# 读取类别文件
with open(args.txt_list, 'r') as file:
    categories = [line.strip() for line in file.readlines()]

# 视角列表
# views = ["front", "side", "top", "back"]

# 设置进度条
pbar = tqdm(total=len(categories)  * args.num_images_per_view, desc="Generating Images")

# 生成图像
for category in categories:
    category_dir = os.path.join(output_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    # for view in views:
    #     view_dir = os.path.join(category_dir, view)
    #     os.makedirs(view_dir, exist_ok=True)
    for idx in range(0, args.num_images_per_view, args.batch_size):
        # 创建提示
        prompts = [f"a Chest X-ray photo of a {category} disease" for _ in range(args.batch_size)]
        # 使用基础模型生成图像
        with torch.inference_mode(), torch.autocast(device.type):
            images = pipe(
                prompt=prompts,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale

            ).images
        
        # 检查生成的图像
        for i, image in enumerate(images):
            image_path = os.path.join(category_dir, f"{idx + i}.png")
            image_np = np.array(image)
            print(f"Image shape: {image_np.shape}")
            print(f"Image min: {image_np.min()}, max: {image_np.max()}, mean: {image_np.mean()}")
            if image_np.min() == 0 and image_np.max() == 0:
                print(f"Warning: Image {image_path} is empty")
            else:
                image.save(image_path)
            del image  # 释放单个图像的显存
            torch.cuda.empty_cache()
            gc.collect()
        
        # 获取已分配的显存（以字节为单位）
        allocated_memory = torch.cuda.memory_allocated(device)
        allocated_memory_gb = allocated_memory / (1024 * 1024 * 1024)
        
        # 计算剩余未使用的显存（以字节为单位）
        free_memory_gb = total_memory_gb - allocated_memory_gb
        
        # 更新进度条
        pbar.set_postfix(**{
            "Image": image_path,
            "Used Mem (GB)": f"{allocated_memory_gb:.2f}",
            "Total Mem (GB)": f"{total_memory_gb:.2f}",
            "Usage": f"{allocated_memory_gb:.2f}GB / {total_memory_gb:.2f}GB"
        })
        pbar.update(args.batch_size)
        torch.cuda.empty_cache()
        gc.collect()

# 完成后关闭进度条
pbar.close()
#python T2I_XL2.py --txt_list ChestX.txt --device cuda:1
