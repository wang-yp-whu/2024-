import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt
from difflib import SequenceMatcher
import os

# 定义图片所在的文件夹路径
folder_path = f'C:\\Users\\PC\\Desktop\\pycode\\question3'

# 去除噪声
def remove_noise(image):
    return cv2.medianBlur(image, 5)

# 加载和预处理图片
def load_and_preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_free_img = remove_noise(gray_img)
    _, binary_img = cv2.threshold(noise_free_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary_img

# OCR文本识别
def ocr_text_extraction(image):
    text = pytesseract.image_to_string(image, lang='chi_sim')
    return text.strip()

# 相似度计算
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# 图片拼接逻辑
def stitch_images(images, order):
    height, width = images[0].shape
    rows = 12  # 假设为12行
    cols = 8   # 假设为8列
    stitched_image = 255 * np.ones((height * rows, width * cols), dtype=np.uint8)

    for idx, img_idx in enumerate(order):
        row_idx = idx // cols
        col_idx = idx % cols
        stitched_image[row_idx * height:(row_idx + 1) * height, col_idx * width:(col_idx + 1) * width] = images[img_idx]

    return stitched_image

# 主程序
def main():
    image_paths = [os.path.join(folder_path, f"{i+1}.bmp") for i in range(96)]
    images = [load_and_preprocess(path) for path in image_paths]
    texts = [ocr_text_extraction(img) for img in images]

    # 简单排序（示例代码，可能需要更复杂的逻辑）
    order = sorted(range(len(texts)), key=lambda k: texts[k])

    # 拼接图片
    stitched_image = stitch_images(images, order)

    # 显示拼接后的图片
    plt.imshow(stitched_image, cmap='gray')
    plt.title("Stitched Image")
    plt.show()

if __name__ == "__main__":
    main()
