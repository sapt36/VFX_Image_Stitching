import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QLabel, QScrollArea, QGridLayout, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sift_impl

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
        disp = cv2.drawKeypoints(disp_img, keypoints, None,
                                 flags=cv2.DrawMatchesFlags_DEFAULT)

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
    window = SIFTVisualizer("test_images_parrington/prtn01.jpg", sigma=1.6, assumed_blur=0.5)
    window.show()
    sys.exit(app.exec_())
