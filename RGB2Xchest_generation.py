import cv2
import numpy as np
import os

# 输入和输出路径
input_directory = r'F:\\Datasets\\\Few-Shot\\mini-imagenet\\images'  # 输入RGB图像文件夹路径
output_directory = r'F:\\Datasets\\Few-Shot\\mini-imagenet\\XChest_style\\'  # 输出XChest风格图像文件夹路径

# 创建输出文件夹（如果不存在的话）
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 遍历输入目录中的所有图像文件
for filename in os.listdir(input_directory):
    file_path = os.path.join(input_directory, filename)

    # 检查文件是否是图像文件（根据扩展名）
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        print(f"Processing: {filename}")
        
        # 读取图像
        img = cv2.imread(file_path)
		
		# 保存RGB图像
        cv2.imwrite(os.path.join(output_directory, f"{filename}_step0_gray.jpg"), img)

        # 1. 将图像转换为灰度图像（模拟X-ray图像的黑白风格）
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 保存步骤1的图像
        cv2.imwrite(os.path.join(output_directory, f"{filename}_step1_gray.jpg"), gray_img)

        # 2. 增加对比度，突出图像中的结构
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))  # 增强对比度
        enhanced_img = clahe.apply(gray_img)
        # 保存步骤2的图像
        cv2.imwrite(os.path.join(output_directory, f"{filename}_step2_enhanced.jpg"), enhanced_img)

        # 3. 使用高斯模糊减少细节，模拟X射线图像的平滑效果
        blurred_img = cv2.GaussianBlur(enhanced_img, (13, 13), 0)
        # 保存步骤3的图像
        cv2.imwrite(os.path.join(output_directory, f"{filename}_step3_blurred.jpg"), blurred_img)

        # 4. 添加适度的噪声，模拟医学影像的伪影或其他特点
        # 生成适度的高斯噪声，以更自然地模拟医学影像中的干扰
        noise = np.random.normal(0, 0.3, blurred_img.shape).astype(np.uint8)  # 标准差降低，噪声更轻微
        noisy_img = cv2.add(blurred_img, noise)  # 通过加权方式更好地融入噪声
        # 保存步骤4的图像
        cv2.imwrite(os.path.join(output_directory, f"{filename}_step4_noisy.jpg"), noisy_img)

        # 最终保存XChest风格图像
        output_path = os.path.join(output_directory, f"{filename}_final_XChest.jpg")
        cv2.imwrite(output_path, noisy_img)

        print(f"Saved XChest style image at: {output_path}")
