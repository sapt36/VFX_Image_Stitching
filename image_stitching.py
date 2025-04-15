import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def harris_corner_detector(img_gray, window_size=3, k=0.04, threshold=1e-5):
    """
    Harris Corner Detector
    img_gray: 灰階影像 (numpy array, 範圍 [0,1])
    window_size: 卷積窗口大小
    k: Harris 公式中的常數
    threshold: 判斷是否為角點的閾值 (可視情況調整)

    回傳：list of (y, x)，表示偵測到角點的位置
    """
    # Step 1: 計算梯度 Ix, Iy
    # 利用簡單 Sobel filter
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)

    Ix = convolve(img_gray, sobel_x)
    Iy = convolve(img_gray, sobel_y)

    # Step 2: 計算 Ix^2, Iy^2, IxIy
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # Step 3: 在窗口內進行加權 (可使用高斯或方形加權)
    # 這裡以簡單的方形窗口做示範
    half_w = window_size // 2

    height, width = img_gray.shape
    R = np.zeros_like(img_gray)

    for r in range(half_w, height - half_w):
        for c in range(half_w, width - half_w):
            M_xx = np.sum(Ix2[r-half_w:r+half_w+1, c-half_w:c+half_w+1])
            M_yy = np.sum(Iy2[r-half_w:r+half_w+1, c-half_w:c+half_w+1])
            M_xy = np.sum(Ixy[r-half_w:r+half_w+1, c-half_w:c+half_w+1])

            # Harris 響應函式 R = det(M) - k * (trace(M)^2)
            detM = (M_xx * M_yy) - (M_xy ** 2)
            traceM = (M_xx + M_yy)
            R[r, c] = detM - k * (traceM ** 2)

    # Step 4: 根據閾值與非極大抑制 (non-maximum suppression) 取出角點
    corners = []
    # 這裡僅以簡易 threshold + 非極大值為示範
    # 真實情況可再加強抑制策略
    for r in range(1, height-1):
        for c in range(1, width-1):
            if R[r, c] > threshold:
                if R[r, c] == np.max(R[r-1:r+2, c-1:c+2]):
                    corners.append((r, c))

    return corners


def convolve(img, kernel):
    """
    簡易 2D 卷積函式，只適用於灰階影像。
    """
    img_h, img_w = img.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2

    # padding
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(img)

    for i in range(img_h):
        for j in range(img_w):
            region = padded[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(region * kernel)
    return output


def extract_descriptor(img_gray, corners, patch_size=8):
    """
    以灰階影像中每個角點 (y, x) 為中心，擷取 patch_size x patch_size 大小的區域作為描述子。
    做簡單的平均/標準化後，拉平成 1D 向量。
    """
    half = patch_size // 2
    descriptors = []
    valid_corners = []
    h, w = img_gray.shape

    for (r, c) in corners:
        if r - half >= 0 and r + half < h and c - half >= 0 and c + half < w:
            patch = img_gray[r-half:r+half, c-half:c+half]
            # 平均為 0，標準差為 1 (簡單正規化)
            patch_mean = np.mean(patch)
            patch_std = np.std(patch) if np.std(patch) != 0 else 1e-6
            norm_patch = (patch - patch_mean) / patch_std
            descriptor = norm_patch.flatten()
            descriptors.append(descriptor)
            valid_corners.append((r, c))
    return np.array(descriptors), valid_corners


def match_features(desc1, desc2, ratio=0.6):
    """
    最簡單的最近鄰搜尋 + ratio test (類似 SIFT Lowe ratio test，簡易版本)。
    desc1: NxD
    desc2: MxD
    ratio: 第一鄰與第二鄰距離比值的閾值

    回傳：配對索引 (idx1, idx2)
    """
    matches = []
    for i, d1 in enumerate(desc1):
        distances = np.linalg.norm(desc2 - d1, axis=1)
        sorted_idx = np.argsort(distances)
        # 最近的兩個
        best1 = sorted_idx[0]
        best2 = sorted_idx[1]
        dist1 = distances[best1]
        dist2 = distances[best2]

        if dist1 / dist2 < ratio:
            matches.append((i, best1))
    return matches


def ransac_homography(pts1, pts2, max_iter=2000, threshold=3.0):
    """
    使用 RANSAC 來估計同形矩陣 H。
    pts1, pts2: shape = (N, 2)，代表配對點的 (x, y)
    max_iter: RANSAC 最大迭代次數
    threshold: 判斷內點的距離閾值

    回傳：最佳 H (3x3) 以及內點集合的布林陣列 mask
    """
    best_inliers = []
    best_H = None
    n = pts1.shape[0]

    if n < 4:
        return np.eye(3), np.ones(n, dtype=bool)  # 至少要 4 點才能估計 H

    for _ in range(max_iter):
        # 隨機抽取 4 點
        idx = np.random.choice(n, 4, replace=False)
        src = pts1[idx]
        dst = pts2[idx]
        # 計算同形矩陣
        H = compute_homography(src, dst)
        # 計算誤差，判斷內點
        if H is None:
            continue

        pts1_h = np.concatenate([pts1, np.ones((n, 1))], axis=1)  # (n, 3)
        projected = (H @ pts1_h.T).T  # (n, 3)
        projected /= (projected[:, 2:3] + 1e-8)

        diff = pts2 - projected[:, :2]
        dist = np.sum(diff**2, axis=1)

        inliers = dist < (threshold**2)
        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_H = H

    return best_H, best_inliers


def compute_homography(src_pts, dst_pts):
    """
    根據配對點 (x1, y1) -> (x2, y2)，解線性方程組求得 3x3 同形矩陣。
    src_pts, dst_pts: shape = (4, 2) or (N, 2), N >= 4
    回傳：H (3x3)
    """
    if src_pts.shape[0] < 4:
        return None

    A = []
    for (x1, y1), (x2, y2) in zip(src_pts, dst_pts):
        A.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])

    A = np.array(A, dtype=np.float64)
    # 透過 SVD 求解
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    # 規範化，使 H[2,2] = 1
    if H[2, 2] != 0:
        H /= H[2, 2]
    return H


def warp_and_blend(img1, img2, H):
    """
    將 img2 warp 到 img1 的座標系之下，並做簡單的拼接/融合。
    這裡採用最簡單的方法，可再改進以避免黑邊或做更好的混合。
    """
    # Step 1: 決定 warp 後的輸出邊界範圍
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img2 = np.array([[0,0],[w2,0],[w2,h2],[0,h2]], dtype=np.float64)
    corners_img2_h = np.concatenate([corners_img2, np.ones((4,1))], axis=1)
    warped_corners = (H @ corners_img2_h.T).T
    warped_corners /= (warped_corners[:, 2:3]+1e-8)  # normalize

    all_x = np.concatenate([warped_corners[:,0], [0, w1]])
    all_y = np.concatenate([warped_corners[:,1], [0, h1]])

    min_x, max_x = int(np.floor(np.min(all_x))), int(np.ceil(np.max(all_x)))
    min_y, max_y = int(np.floor(np.min(all_y))), int(np.ceil(np.max(all_y)))

    # 新影像的大小
    out_w = max_x - min_x
    out_h = max_y - min_y

    # Step 2: 建立結果影像的座標網格
    xx, yy = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    ones = np.ones_like(xx_flat, dtype=np.float64)
    coords = np.vstack((xx_flat, yy_flat, ones))

    # Step 3: 將 output 座標反轉回去 img1, img2 原座標
    # 對於 img1，因為它相當於參考平面，所以不需要轉換。
    # 對於 img2，需要用 H^-1 進行反向映射。
    out_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # 先把 img1 放進新座標
    # 避免超出邊界，可以直接用條件判斷
    offset_x = -min_x
    offset_y = -min_y
    out_img[offset_y:offset_y+h1, offset_x:offset_x+w1] = img1

    # warp img2
    H_inv = np.linalg.inv(H)
    warp_coords = (H_inv @ coords).T
    warp_coords /= (warp_coords[:, 2:3] + 1e-8)

    warp_x = warp_coords[:, 0].round().astype(int)
    warp_y = warp_coords[:, 1].round().astype(int)

    valid_mask = (warp_x >= 0) & (warp_x < w2) & (warp_y >= 0) & (warp_y < h2)
    out_x = xx_flat + offset_x
    out_y = yy_flat + offset_y

    # 簡易混合：若 out_img 有黑色(0,0,0)，就用 img2 像素填；否則保留。
    for i in range(len(valid_mask)):
        if valid_mask[i]:
            vx = warp_x[i]
            vy = warp_y[i]
            ox = out_x[i]
            oy = out_y[i]
            if np.all(out_img[oy, ox] == 0):
                out_img[oy, ox] = img2[vy, vx]

    return out_img


def main():
    # 載入影像 (請換成你自己的路徑或測試影像)
    img1_path = "test_images/prtn00.jpg"
    img2_path = "test_images/prtn01.jpg"

    img1 = np.array(Image.open(img1_path).convert('RGB'))
    img2 = np.array(Image.open(img2_path).convert('RGB'))

    # 轉灰階
    gray1 = np.array(Image.open(img1_path).convert('L'), dtype=np.float32) / 255.0
    gray2 = np.array(Image.open(img2_path).convert('L'), dtype=np.float32) / 255.0

    # 1. 特徵偵測 (Harris Corner)
    corners1 = harris_corner_detector(gray1, window_size=3, k=0.04, threshold=1e-5)
    corners2 = harris_corner_detector(gray2, window_size=3, k=0.04, threshold=1e-5)

    # 2. 特徵描述 (擷取 patch + 正規化)
    desc1, valid_corners1 = extract_descriptor(gray1, corners1, patch_size=8)
    desc2, valid_corners2 = extract_descriptor(gray2, corners2, patch_size=8)

    # 3. 特徵匹配 (最近鄰 + ratio test)
    matches = match_features(desc1, desc2, ratio=0.7)

    # 4. 利用 RANSAC 估計 Homography
    pts1 = []
    pts2 = []
    for i1, i2 in matches:
        y1, x1 = valid_corners1[i1]
        y2, x2 = valid_corners2[i2]
        pts1.append([x1, y1])  # (x, y)
        pts2.append([x2, y2])

    pts1 = np.array(pts1, dtype=np.float64)
    pts2 = np.array(pts2, dtype=np.float64)

    H, inliers = ransac_homography(pts1, pts2, max_iter=2000, threshold=3.0)
    print("Homography = \n", H)

    # 5. 影像變形 (Warp) + 6. 影像融合 (簡易拼接)
    result = warp_and_blend(img1, img2, H)

    # 顯示結果
    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.title("Image 1")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.title("Image 2")
    plt.axis('off')

    plt.figure()
    plt.imshow(result)
    plt.title("Stitched Result")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
