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
import sift_impl
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)


def cvimg_to_qpixmap(img, max_width=None, max_height=None):
    """將 OpenCV 影像 (灰階或 BGR) 轉為 Qt QPixmap，並可選擇縮放至最大尺寸"""
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    if len(img.shape) == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
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

    :param img:         原始影像（BGR、RGB 或灰階）
    :param keypoints:   cv2.KeyPoint 列表
    :param point_color: 特徵點的顏色 (Matplotlib 顏色名稱)
    :param arrow_color: 方向箭頭顏色 (Matplotlib 顏色名稱)
    :param scale:       箭頭長度縮放倍率
    :return:            disp (ndarray)，包含繪製結果的影像（BGR格式）
    """
    # 1) 若為灰階，轉為 RGB；若是 BGR，也轉為 RGB 以正確顯示色彩 (Matplotlib預設視為RGB)
    if len(img.shape) == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # 假設第三維不是3則可能已是RGB或其他情況，簡單做copy
        img_rgb = img.copy()

    # 2) 建立 figure，並在其中繪製影像與特徵點
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img_rgb)
    ax.set_axis_off()  # 不顯示軸線

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

    # 3) 使 Matplotlib 將圖渲染到快取（非顯示到螢幕），再讀回成陣列
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()  # 取得圖像寬高(像素)
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(h, w, 3)  # 形狀 (height, width, 3)，RGB 格式

    # 4) Matplotlib 繪製完後，可關閉 figure
    plt.close(fig)

    # 5) 將 RGB 轉為 BGR，以便後續可用 cv2.imshow 或 cv2.imwrite 等
    disp = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    return disp


class SIFTVisualizer(QMainWindow):
    def __init__(self, image_path, sigma=1.6, assumed_blur=0.5):
        super().__init__()
        self.sigma = sigma
        self.assumed_blur = assumed_blur
        # 計算 SIFT 流程
        self.orig_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = self.orig_image.astype('float32')

        self.base_image = sift_impl.generate_base_image(img, sigma=self.sigma, assumed_blur=self.assumed_blur)
        num_octaves = sift_impl.compute_number_of_octaves(self.base_image.shape)
        kernels = sift_impl.generate_gaussian_kernels(self.sigma, num_intervals=3)
        self.gaussian_images = sift_impl.generate_gaussian_images(self.base_image, num_octaves, kernels)
        self.dog_images = sift_impl.generate_DoG_images(self.gaussian_images)
        raw_kp = sift_impl.find_scale_space_extrema(self.gaussian_images, self.dog_images, num_intervals=3, sigma=self.sigma, border=5)
        no_dup = sift_impl.remove_duplicate_keypoints(raw_kp)
        self.kp_converted = sift_impl.convert_keypoints_to_input_image_size(no_dup)
        self.descriptors = sift_impl.generate_descriptors(self.kp_converted, self.gaussian_images)
        self.raw_kp = raw_kp

        # 建立分頁
        tabs = QTabWidget()
        tabs.addTab(self.create_image_tab(self.base_image, "Base Image"), "Base Image")
        tabs.addTab(self.create_pyramid_tab(self.gaussian_images[0], "Gaussian Pyramid"), "Gaussian Pyramid")
        tabs.addTab(self.create_pyramid_tab(self.dog_images[0], "DoG Pyramid"), "DoG Pyramid")
        tabs.addTab(
            self.create_keypoints_tab(self.kp_converted, "Converted Keypoints", self.orig_image),
            "Converted Keypoints"
        )
        tabs.addTab(self.create_descriptor_tab(self.descriptors), "Descriptor Vector")

        self.setCentralWidget(tabs)
        self.setWindowTitle("SIFT Process Visualizer")
        self.setFont(QFont('Times New Roman', 12, QFont.Bold))
        self.resize(1024, 768)  # 初始視窗大小，適合 1920x1080 顯示器

    def create_image_tab(self, img, title):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        # 顯示標題與參數
        title_lbl = QLabel(f"{title}\nσ={self.sigma}, assumed_blur={self.assumed_blur}")
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)
        # 顯示影像
        label = QLabel()
        pix = cvimg_to_qpixmap(img, max_width=800, max_height=600)
        label.setPixmap(pix)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        # 拉伸空間
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

        # 顯示標題、參數、以及鍵點數
        num_kp = len(keypoints)  # 取得鍵點數量
        title_lbl = QLabel(
            f"{title}\n"
            f"σ={self.sigma}, assumed_blur={self.assumed_blur}\n"
            f"Keypoints: {num_kp}"
        )
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)

        # 畫鍵點 (若img沒指定，就用 self.base_image)
        draw_img = img if img is not None else self.base_image
        # 防止 draw_img 為 float或過大時無法正常畫
        disp_img = (draw_img / np.max(draw_img) * 255).astype('uint8')
        disp = draw_feature_points_return_disp(disp_img, keypoints, scale=3)

        # 顯示鍵點影像
        label = QLabel()
        pix = cvimg_to_qpixmap(disp, max_width=800, max_height=600)
        label.setPixmap(pix)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        layout.addStretch()
        return widget

    def create_descriptor_tab(self, descriptors):
        fig = Figure(figsize=(6,3))
        ax = fig.add_subplot(111)
        if descriptors.shape[0] > 0:
            ax.bar(range(descriptors.shape[1]), descriptors[0])
            ax.set_title("First Descriptor Vector")
        canvas = FigureCanvas(fig)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(canvas)
        return widget

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SIFTVisualizer("parrington/prtn01.jpg", sigma=1.6, assumed_blur=0.5)
    window.show()
    sys.exit(app.exec_())
