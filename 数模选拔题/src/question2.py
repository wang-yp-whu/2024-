import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from difflib import SequenceMatcher
import os

# 定义图片所在的文件夹路径
folder_path = f'question2'


# 加载和预处理图片
def load_and_preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary_img


# OCR文本识别
def ocr_text_extraction(image):
    text = pytesseract.image_to_string(image, lang='chi_sim')  #使用中文简体模型
    return text.strip()


# 相似度计算
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# 图片拼接逻辑
def stitch_images(images, order):
    # 假设所有图片大小相同
    height, width = images[0].shape
    rows = 12  # 根据问题描述，我们假设纵横拼接分别为12行
    cols = int(len(images) / rows)
    stitched_image = 255 * np.ones((height * rows, width * cols), dtype=np.uint8)  # 创建大型白底图像

    for idx, img_idx in enumerate(order):
        row_idx = idx // cols
        col_idx = idx % cols
        stitched_image[row_idx * height:(row_idx + 1) * height, col_idx * width:(col_idx + 1) * width] = images[img_idx]

    return stitched_image


# 主程序
def main():
    # 获取文件夹内所有图片文件
    image_paths = [[f'question2/{i}.bmp' for i in range(1, 97)]]
    images = [load_and_preprocess(path) for path in image_paths]
    texts = [ocr_text_extraction(img) for img in images]

    # 创建相似度矩阵并简单排序（示例代码，需要更复杂的逻辑）
    order = sorted(range(len(texts)), key=lambda k: texts[k])

    # 拼接图片
    stitched_image = stitch_images(images, order)

    # 显示拼接后的图片
    plt.imshow(stitched_image, cmap='gray')
    plt.title("Stitched Image")
    plt.show()


if __name__ == "__main__":
    main()
