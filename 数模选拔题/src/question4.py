import cv2
import numpy as np


def stitch_images(img1, img2):
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 对两张图片分别找到关键点和描述子
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 使用BFMatcher进行匹配，并进行KNN筛选
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # 应用比率测试找到好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 至少需要4个好的匹配点来找到变换矩阵
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 使用RANSAC方法找到一个透视变换
        matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 使用变换矩阵对img1进行透视变换
        height, width, channels = img2.shape
        result = cv2.warpPerspective(img1, matrix, (width, height))

        # 将img2叠加到变换后的img1上
        result[0:height, 0:width] = img2
        return result
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), 4))
        return None


# 加载图片
# stitch_image = [f'question4/{i}.bmp' for i in range(1, 121)]
# image_path = f'question4/grassland.JPG'
# img = cv2.imread(image_path)
# cv2.imshow('Image', img)
# cv2.waitKey(0)  # 等待直到用户按键
# cv2.destroyAllWindows()

# 显示拼接结果
# if stitched_image is not None:
#     cv2.imshow('Stitched Image', stitched_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Image stitching failed.")
import cv2


def display_image(image_path):
    # 加载图片
    img = cv2.imread(image_path)

    # 检查图片是否正确加载
    if img is not None:
        # 调整图片尺寸
        # 可以自定义尺寸或按比例缩放
        height, width = img.shape[:2]
        # 例如，缩放到屏幕分辨率的50%
        img = cv2.resize(img, (width // 4, height // 4))

        # 显示图片
        cv2.imshow('Resized Grassland Image', img)
        cv2.waitKey(0)  # 等待直到用户按键
        cv2.destroyAllWindows()  # 关闭所有窗口
    else:
        print("Error: Image could not be loaded.")


# 调用函数，确保替换为你的图片路径
display_image('C:\\Users\\PC\\Desktop\\pycode\\question4\\grassland.JPG')
