import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QLabel, QScrollArea, QGridLayout
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sift_impl

#####################################################
# 1) 影像/圖像處理的工具函式
#####################################################
def cvimg_to_qpixmap(img, max_width=None, max_height=None):
    """將 OpenCV 影像 (灰階或 BGR) 轉為 Qt QPixmap，並可選擇縮放至最大尺寸"""
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    if len(img.shape) == 2:
        # 灰階
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        # BGR -> RGB
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)

    pix = QPixmap.fromImage(qimg.copy())
    # 如果指定了最大尺寸，按比例縮放
    if max_width or max_height:
        pix = pix.scaled(
            max_width or pix.width(),
            max_height or pix.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
    return pix

def draw_feature_points_return_disp(img, keypoints, point_color='red', arrow_color='yellow', scale=0.5):
    """
    回傳一張包含特徵點的可視化影像 (disp)，而非直接顯示到螢幕上。
    """
    # 若為灰階 -> RGB；若為 BGR -> 轉為 RGB 以正確顯示色彩
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img.copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img_rgb)
    ax.set_axis_off()

    for kp in keypoints:
        x, y = kp.pt
        # 畫出特徵點的位置
        ax.plot(x, y, 'o', color=point_color, markersize=2)

        # 若 keypoint 有方向資訊，則畫箭頭
        if kp.angle != -1:
            angle_rad = np.deg2rad(kp.angle)
            dx = np.cos(angle_rad) * kp.size / scale
            dy = np.sin(angle_rad) * kp.size / scale
            ax.arrow(x, y, dx, dy,
                     color=arrow_color,
                     head_width=1.5, head_length=2)

    ax.set_title("Feature Points with Orientation")
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(h, w, 3)
    plt.close(fig)

    disp = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    return disp

#####################################################
# 2) PyQt 主體視窗：SIFTVisualizer
#####################################################
class SIFTVisualizer(QMainWindow):
    def __init__(self, image_path, sigma=1.6, assumed_blur=0.5):
        super().__init__()
        self.sigma = sigma
        self.assumed_blur = assumed_blur

        # 讀取與準備影像
        self.orig_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = self.orig_image.astype('float32')

        # ---------------------
        # 以下為 SIFT 實作流程
        # ---------------------
        self.base_image = sift_impl.generate_base_image(img, sigma=self.sigma, assumed_blur=self.assumed_blur)
        num_octaves = sift_impl.compute_number_of_octaves(self.base_image.shape)
        kernels = sift_impl.generate_gaussian_kernels(self.sigma, num_intervals=3)
        self.gaussian_images = sift_impl.generate_gaussian_images(self.base_image, num_octaves, kernels)
        self.dog_images = sift_impl.generate_DoG_images(self.gaussian_images)

        # 找特徵點後做重複排除、再轉回輸入影像大小
        raw_kp = sift_impl.find_scale_space_extrema(self.gaussian_images, self.dog_images,
                                                    num_intervals=3, sigma=self.sigma, border=5)
        no_dup = sift_impl.remove_duplicate_keypoints(raw_kp)
        self.kp_converted = sift_impl.convert_keypoints_to_input_image_size(no_dup)
        self.descriptors = sift_impl.generate_descriptors(self.kp_converted, self.gaussian_images)
        self.raw_kp = raw_kp

        # ---------------------
        # 建立分頁 (QTabWidget)
        # ---------------------
        tabs = QTabWidget()
        # (1) Base Image
        tabs.addTab(self.create_image_tab(self.base_image, "Base Image"), "Base Image")
        # (2) Gaussian Pyramid
        tabs.addTab(self.create_pyramid_tab(self.gaussian_images[0], "Gaussian Pyramid"), "Gaussian Pyramid")
        # (3) DoG Pyramid
        tabs.addTab(self.create_pyramid_tab(self.dog_images[0], "DoG Pyramid"), "DoG Pyramid")
        # (4) Keypoints Visualization
        tabs.addTab(
            self.create_keypoints_tab(self.kp_converted, "Converted Keypoints", self.orig_image),
            "Converted Keypoints"
        )
        # (5) Descriptors
        tabs.addTab(self.create_descriptor_tab(self.descriptors), "Descriptor Vector")
        # (6) Feature Matching (可選擇matching對象)
        tabs.addTab(
            self.create_feature_matching_tab("parrington/prtn01.jpg", "parrington/prtn00.jpg"),
            "Feature Matching"
        )

        self.setCentralWidget(tabs)
        self.setWindowTitle("SIFT Process Visualizer")
        self.setFont(QFont('Times New Roman', 12, QFont.Bold))
        self.resize(1024, 768)  # 初始視窗大小

    #####################################################
    # 分頁頁面函式區
    #####################################################
    def create_image_tab(self, img, title):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title_lbl = QLabel(f"{title}\nσ={self.sigma}, assumed_blur={self.assumed_blur}")
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)

        label = QLabel()
        pix = cvimg_to_qpixmap(img, max_width=800, max_height=600)
        label.setPixmap(pix)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        layout.addStretch()
        return widget

    def create_pyramid_tab(self, images, title):
        container = QWidget()
        grid = QGridLayout(container)
        thumb_width = 250
        thumb_height = 180
        cols = 3

        for idx, im in enumerate(images):
            text = QLabel(f"{title} Level {idx}")
            text.setAlignment(Qt.AlignCenter)

            pix = cvimg_to_qpixmap(im, max_width=thumb_width, max_height=thumb_height)
            img_lbl = QLabel()
            img_lbl.setPixmap(pix)
            img_lbl.setAlignment(Qt.AlignCenter)

            vbox = QVBoxLayout()
            vbox.addWidget(text)
            vbox.addWidget(img_lbl)
            cell = QWidget()
            cell.setLayout(vbox)
            grid.addWidget(cell, idx // cols, idx % cols)

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        return scroll

    def create_keypoints_tab(self, keypoints, title, img=None):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        num_kp = len(keypoints)
        title_lbl = QLabel(
            f"{title}\n"
            f"σ={self.sigma}, assumed_blur={self.assumed_blur}\n"
            f"Keypoints: {num_kp}"
        )
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)

        draw_img = img if img is not None else self.base_image
        disp_img = (draw_img / np.max(draw_img) * 255).astype('uint8')
        disp = draw_feature_points_return_disp(disp_img, keypoints, scale=3)

        label = QLabel()
        pix = cvimg_to_qpixmap(disp, max_width=800, max_height=600)
        label.setPixmap(pix)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        layout.addStretch()
        return widget

    def create_descriptor_tab(self, descriptors):
        fig = Figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        if descriptors.shape[0] > 0:
            ax.bar(range(descriptors.shape[1]), descriptors[0])
            ax.set_title("First Descriptor Vector")

        canvas = FigureCanvas(fig)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(canvas)
        return widget

    def create_feature_matching_tab(self, query_path, train_path):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        title_lbl = QLabel("Feature Matching with FLANN + Homography")
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)

        img1 = cv2.imread(query_path, 0)  # query
        img2 = cv2.imread(train_path, 0)  # train

        kp1, des1 = sift_impl.compute_keypoints_and_descriptors(img1)
        kp2, des2 = sift_impl.compute_keypoints_and_descriptors(img2)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        MIN_MATCH_COUNT = 10
        info_label = QLabel()
        info_label.setAlignment(Qt.AlignCenter)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h, w = img1.shape
            pts = np.float32([[0, 0],
                              [0, h - 1],
                              [w - 1, h - 1],
                              [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            cv2.polylines(img2_color, [np.int32(dst)], True, (255, 255, 255), 3, cv2.LINE_AA)

            h1, w1 = img1.shape
            h2, w2 = img2_color.shape[:2]
            nWidth = w1 + w2
            nHeight = max(h1, h2)
            hdif = int((h2 - h1) / 2)

            newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
            img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            newimg[hdif:hdif + h1, :w1, :] = img1_color
            newimg[:h2, w1:w1 + w2, :] = img2_color

            for m in good:
                pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
                pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
                cv2.line(newimg, pt1, pt2, (255, 0, 0), 1)

            info_label.setText(f"Found {len(good)} good matches (MIN_MATCH_COUNT={MIN_MATCH_COUNT})")
            final_pix = cvimg_to_qpixmap(newimg, max_width=900, max_height=600)
        else:
            info_label.setText(f"Not enough matches found: {len(good)}/{MIN_MATCH_COUNT}")
            color_img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            final_pix = cvimg_to_qpixmap(color_img2, max_width=900, max_height=600)

        layout.addWidget(info_label)

        match_label = QLabel()
        match_label.setPixmap(final_pix)
        match_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(match_label)

        layout.addStretch()
        return widget

#####################################################
# 3) 主程式進入點
#####################################################
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 以 prtn01.jpg 示例
    window = SIFTVisualizer("parrington/prtn01.jpg", sigma=1.6, assumed_blur=0.5)
    window.show()
    sys.exit(app.exec_())
