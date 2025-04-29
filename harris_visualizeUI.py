import sys
import os
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QSize


#############################
# 1) Harris + Descriptor + Matching 核心函式
#############################

def conv2d(img, kernel):
    """2D 捲積函數 (簡化版)"""
    h, w = img.shape
    m, n = kernel.shape
    pad_img = np.pad(img, (m // 2, n // 2), 'edge').astype(np.float64)
    result = np.zeros_like(img, dtype=np.float64)
    for i in range(m):
        for j in range(n):
            result += pad_img[i:i + h, j:j + w] * kernel[i, j]
    return result

def HarrisCorner(img_bgr, max_points=200, k=0.05, block_size=21, gauss_sigma=2, thresh_ratio=0.02):
    """
    Harris Corner檢測
    回傳： (corner_candidates, Ix, Iy)
    corner_candidates 為 [(y, x, R), ...]
    """
    h, w, _ = img_bgr.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 計算 Ix, Iy (簡單卷積)
    Hx = np.array([[0, 0, 0],
                   [1, 0, -1],
                   [0, 0, 0]], dtype=np.float32)
    Hy = np.array([[0, 1, 0],
                   [0, 0, 0],
                   [0, -1, 0]], dtype=np.float32)

    Ix = conv2d(gray, Hx)
    Iy = conv2d(gray, Hy)

    # 高斯平滑
    Ix2 = cv2.GaussianBlur(Ix**2, (block_size, block_size), gauss_sigma)
    Iy2 = cv2.GaussianBlur(Iy**2, (block_size, block_size), gauss_sigma)
    Ixy = cv2.GaussianBlur(Ix * Iy, (block_size, block_size), gauss_sigma)

    # Harris R
    detM = (Ix2 * Iy2) - (Ixy**2)
    traceM = Ix2 + Iy2
    R = detM - k * (traceM**2)

    # threshold
    harris_max = np.max(R)
    threshold = harris_max * thresh_ratio
    corner_candidates = []
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if R[i, j] > threshold:
                local_patch = R[i - 1:i + 2, j - 1:j + 2]
                if R[i, j] == np.max(local_patch):
                    corner_candidates.append((i, j, R[i, j]))

    corner_candidates.sort(key=lambda x: x[2], reverse=True)
    corner_candidates = corner_candidates[:max_points]
    return corner_candidates, Ix, Iy

def calc_orientation(Ix, Iy):
    """計算梯度大小 m 與角度 theta (0~360)"""
    m = np.sqrt(Ix**2 + Iy**2)
    theta = np.rad2deg(np.arctan2(Iy, Ix)) % 360
    return m, theta

def gen_descriptor(fpx, fpy, m, theta):
    """
    建立 16x16 patch 的 SIFT-like descriptor，回傳128維向量
    """
    pad_size = 8
    m_padded = np.pad(m, pad_size, mode='edge')
    theta_padded = np.pad(theta, pad_size, mode='edge')
    fpx_ = fpx + pad_size
    fpy_ = fpy + pad_size

    patchM = m_padded[fpx_:fpx_ + 16, fpy_:fpy_ + 16].copy()
    patchT = theta_padded[fpx_:fpx_ + 16, fpy_:fpy_ + 16].copy()

    patchM = cv2.GaussianBlur(patchM, (9, 9), 4.5)
    bins = 8
    angles_hist = np.zeros((bins,), dtype=np.float32)

    for i in range(16):
        for j in range(16):
            mag_val = patchM[i, j]
            ang_val = patchT[i, j] % 360
            idx_bin = int((ang_val / 360) * bins) % bins
            angles_hist[idx_bin] += mag_val

    main_theta = (np.argmax(angles_hist) + 0.5) * (360 / bins)

    patchT -= main_theta
    patchT = (patchT + 360) % 360

    descriptor = []
    block_size = 4
    for by in range(4):
        for bx in range(4):
            subM = patchM[by*block_size:(by+1)*block_size, bx*block_size:(bx+1)*block_size]
            subT = patchT[by*block_size:(by+1)*block_size, bx*block_size:(bx+1)*block_size]
            hist_8 = np.zeros((bins,), dtype=np.float32)
            for yy in range(block_size):
                for xx in range(block_size):
                    mag_val = subM[yy, xx]
                    ang_val = subT[yy, xx] % 360
                    idx_bin = int((ang_val / 360) * bins) % bins
                    hist_8[idx_bin] += mag_val
            descriptor.extend(hist_8)
    descriptor = np.array(descriptor, dtype=np.float32)
    descriptor /= (np.linalg.norm(descriptor) + 1e-7)
    descriptor = np.clip(descriptor, 0, 0.2)
    descriptor /= (np.linalg.norm(descriptor) + 1e-7)
    return descriptor

def compute_keypoints_and_descriptors_harris(img_bgr, max_points=200):
    """
    Harris + Descriptor
    回傳:
      kps: [ (x, y), ... ]
      descs: ndarray shape=(N, 128)
    """
    corners, Ix, Iy = HarrisCorner(img_bgr, max_points=max_points)
    m, theta = calc_orientation(Ix, Iy)

    h, w = img_bgr.shape[:2]
    margin = 8
    kps = []
    descs = []

    for (yy, xx, val) in corners:
        if yy < margin or yy >= h - margin:
            continue
        if xx < margin or xx >= w - margin:
            continue
        kps.append((xx, yy))
        desc = gen_descriptor(yy, xx, m, theta)
        descs.append(desc)

    descs = np.array(descs, dtype=np.float32)
    return kps, descs

def simple_match(kpsA, descA, kpsB, descB, desc_thresh=1.0):
    """最近鄰匹配: L2距離 < desc_thresh"""
    matches = []
    for i in range(len(descA)):
        best_dist = float('inf')
        best_idx = -1
        vecA = descA[i]
        for j in range(len(descB)):
            diff = vecA - descB[j]
            dist = np.dot(diff, diff)
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        if best_dist < desc_thresh:
            matches.append((kpsA[i], kpsB[best_idx]))
    return matches


#############################
# 2) PyQt5 介面
#############################
def convertCV2Qt(img_bgr):
    """
    將 OpenCV BGR 格式的 numpy array 轉成 PyQt 可顯示的 QPixmap
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytesPerLine = ch * w
    qimg = QImage(img_rgb.data, w, h, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def draw_harris_corners_on_image(img_bgr, keypoints):
    """
    在影像上畫出 Harris 角點，回傳新影像 (copy)
    keypoints: [(x, y), ...]
    """
    out = img_bgr.copy()
    for (x, y) in keypoints:
        cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1)  # 紅色圓點
    return out

def draw_matches_side_by_side(imgA, kpsA, imgB, kpsB, matches):
    """
    將兩張影像水平拼接後畫出匹配線
    matches: [((xA,yA),(xB,yB)), ...]
    回傳: 拼接後的影像
    """
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]
    H = max(hA, hB)
    W = wA + wB

    merged = np.zeros((H, W, 3), dtype=np.uint8)
    merged[:hA, :wA] = imgA
    merged[:hB, wA:wA + wB] = imgB

    # 在 merged 上用 line 表示匹配
    for (ptA, ptB) in matches:
        xA, yA = ptA
        xB, yB = ptB
        p1 = (int(xA), int(yA))
        p2 = (int(xB + wA), int(yB))

        color_line = (0, 255, 0)
        cv2.line(merged, p1, p2, color_line, 1)
        cv2.circle(merged, p1, 4, (0, 0, 255), -1)
        cv2.circle(merged, p2, 4, (255, 0, 0), -1)

    return merged


class HarrisDemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Harris Corner Dectector")
        self.setFont(QFont('Times New Roman', 12, QFont.Bold))
        self.setFixedSize(1200, 700)

        # ----- UI Layout -----
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 影像顯示區 (左右各一張)
        self.label_imgA = QLabel("Image A")
        self.label_imgA.setFixedSize(500, 300)
        self.label_imgA.setStyleSheet("background-color: #cccccc;")
        self.label_imgA.setScaledContents(True)

        self.label_imgB = QLabel("Image B")
        self.label_imgB.setFixedSize(500, 300)
        self.label_imgB.setStyleSheet("background-color: #cccccc;")
        self.label_imgB.setScaledContents(True)

        # 匹配結果顯示
        self.label_matches = QLabel("Matches")
        self.label_matches.setFixedSize(1000, 300)
        self.label_matches.setStyleSheet("background-color: #dddddd;")
        self.label_matches.setScaledContents(True)

        # 按鈕
        self.btn_loadA = QPushButton("Load Image A")
        self.btn_loadA.clicked.connect(self.load_imageA)
        self.btn_loadB = QPushButton("Load Image B")
        self.btn_loadB.clicked.connect(self.load_imageB)
        self.btn_run = QPushButton("Harris Dectection + Matching")
        self.btn_run.clicked.connect(self.runHarrisAndMatch)

        # Layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.label_imgA)
        top_layout.addWidget(self.label_imgB)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.label_matches)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_loadA)
        btn_layout.addWidget(self.btn_loadB)
        btn_layout.addWidget(self.btn_run)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(bottom_layout)

        main_widget.setLayout(main_layout)

        # ----- 資料 -----
        self.imgA = None
        self.imgB = None

    def load_imageA(self):
        fname, _ = QFileDialog.getOpenFileName(self, "選取影像 A", ".", "Images (*.png *.jpg *.bmp)")
        if fname:
            self.imgA = cv2.imread(fname)
            if self.imgA is None:
                QMessageBox.warning(self, "讀取失敗", f"無法讀取檔案: {fname}")
                return
            pixmapA = convertCV2Qt(self.imgA)
            self.label_imgA.setPixmap(pixmapA)

    def load_imageB(self):
        fname, _ = QFileDialog.getOpenFileName(self, "選取影像 B", ".", "Images (*.png *.jpg *.bmp)")
        if fname:
            self.imgB = cv2.imread(fname)
            if self.imgB is None:
                QMessageBox.warning(self, "讀取失敗", f"無法讀取檔案: {fname}")
                return
            pixmapB = convertCV2Qt(self.imgB)
            self.label_imgB.setPixmap(pixmapB)

    def runHarrisAndMatch(self):
        if self.imgA is None or self.imgB is None:
            QMessageBox.information(self, "提醒", "請先載入影像 A 與 B")
            return

        # --- Harris Corner ---
        kpsA, descA = compute_keypoints_and_descriptors_harris(self.imgA, max_points=200)
        kpsB, descB = compute_keypoints_and_descriptors_harris(self.imgB, max_points=200)

        # 在影像上畫出角點
        showA = draw_harris_corners_on_image(self.imgA, kpsA)
        showB = draw_harris_corners_on_image(self.imgB, kpsB)

        self.label_imgA.setPixmap(convertCV2Qt(showA))
        self.label_imgB.setPixmap(convertCV2Qt(showB))

        # --- Matching ---
        matches = simple_match(kpsA, descA, kpsB, descB, desc_thresh=1.0)

        # 拼接並畫出匹配線
        match_img = draw_matches_side_by_side(self.imgA, kpsA, self.imgB, kpsB, matches)
        self.label_matches.setPixmap(convertCV2Qt(match_img))


def main():
    app = QApplication(sys.argv)
    window = HarrisDemoWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
