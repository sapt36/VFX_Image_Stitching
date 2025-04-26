import os
import cv2
import numpy as np
import math
import time

#############################
# 0) pano.txt 讀取
#############################
def read_pano_data(pano_file_path):
    """
    讀取 pano.txt 資料 (made by AutoStitch)

    流程：
      1. 尋找包含 .jpg 或 .png 的行 → 當作影像路徑
      2. 下一行若可轉 float → 作為該影像的焦距(像素)
      3. 其餘行 (包含空白或矩陣) 皆略過
    回傳：
      images  (list): 影像路徑清單
      focuses (list): 焦距數值清單
    """
    images = []
    focuses = []
    pending_img = None

    with open(pano_file_path, 'r', encoding='utf-8') as f:
        all_lines = f.read().splitlines()

    for text_line in all_lines:
        line_stripped = text_line.strip().lower()
        if ('.jpg' in line_stripped) or ('.png' in line_stripped):
            pending_img = text_line.strip()  # 暫存完整字串(可能大小寫混合)
        else:
            # 若此行可能是焦距 (float)
            if (' ' not in line_stripped) and (len(line_stripped) > 0):
                try:
                    val = float(line_stripped)
                    if pending_img is not None:
                        images.append(pending_img)
                        focuses.append(val)
                        pending_img = None
                except ValueError:
                    pass
    return images, focuses

#############################
# (A) Con2D & Harris Corner Detector
#############################
def conv2d(img, kernel):
    """
    2D 捲積函數
    """
    h, w = img.shape
    m, n = kernel.shape
    # 以 edge 模式 padding
    pad_img = np.pad(img, (m // 2, n // 2), 'edge').astype(np.float64)
    result = np.zeros_like(img, dtype=np.float64)
    for i in range(m):
        for j in range(n):
            result += pad_img[i:i + h, j:j + w] * kernel[i, j]
    return result

def calc_orientation(Ix, Iy):
    """
    計算梯度大小 m 與角度 theta (0~360)
    """
    m = np.sqrt(Ix**2 + Iy**2)
    theta = np.arctan2(Iy, Ix) * 180 / np.pi
    theta = (theta + 360) % 360
    return m, theta

def gen_descriptor(fpx, fpy, m, theta):
    """
    以 16x16 patch，分割成4x4小塊，每小塊計 8-bin，總共4*4*8=128維描述子 (類似 SIFT 方向直方圖概念)。
    """
    # 為了安全，對 m, theta 做 padding (避免index超界)
    # 先 pad(8,8)，讓 fpx+8, fpy+8 可以安全取到 16x16
    pad_size = 8
    m_padded = np.pad(m, pad_size, mode='edge')
    theta_padded = np.pad(theta, pad_size, mode='edge')
    # 在原圖對應位置就要偏移 pad_size
    fpx_ = fpx + pad_size
    fpy_ = fpy + pad_size

    # 取 16x16 patch
    patchM = m_padded[fpx_:fpx_ + 16, fpy_:fpy_ + 16].copy()
    patchT = theta_padded[fpx_:fpx_ + 16, fpy_:fpy_ + 16].copy()

    # 找主要方向(類似 SIFT main orientation)
    # 先用 9x9(或 16x16) Gaussian 做平滑 (此處簡化也可)
    patchM = cv2.GaussianBlur(patchM, (9, 9), 1.5 * 3)
    bins = 8
    angles_hist = np.zeros((bins,), dtype=np.float32)

    # 計算整個 16x16 patch 的方向直方圖
    for i in range(16):
        for j in range(16):
            mag_val = patchM[i, j]
            ang_val = patchT[i, j] % 360
            idx_bin = int((ang_val / 360) * bins) % bins
            angles_hist[idx_bin] += mag_val

    # 主方向
    main_theta = (np.argmax(angles_hist) + 0.5) * (360 / bins)

    # 重新計算 patchT - main_theta
    patchT -= main_theta
    patchT = (patchT + 360) % 360

    # 分成 4x4 區塊，每區計算8-bin直方圖
    descriptor = []
    block_size = 4
    for by in range(4):
        for bx in range(4):
            subM = patchM[by * block_size:(by + 1) * block_size,
                          bx * block_size:(bx + 1) * block_size]
            subT = patchT[by * block_size:(by + 1) * block_size,
                          bx * block_size:(bx + 1) * block_size]
            hist_8 = np.zeros((bins,), dtype=np.float32)
            for yy in range(block_size):
                for xx in range(block_size):
                    mag_val = subM[yy, xx]
                    ang_val = subT[yy, xx] % 360
                    idx_bin = int((ang_val / 360) * bins) % bins
                    hist_8[idx_bin] += mag_val
            descriptor.extend(hist_8)
    descriptor = np.array(descriptor, dtype=np.float32)

    # 正規化 (類似 SIFT，限制 <= 0.2，再二次正規化)
    descriptor /= (np.linalg.norm(descriptor) + 1e-7)
    descriptor = np.clip(descriptor, 0, 0.2)
    descriptor /= (np.linalg.norm(descriptor) + 1e-7)
    return descriptor

def HarrisCorner(img_bgr, max_points=200, k=0.05, block_size=21, gauss_sigma=2, thresh_ratio=0.02):
    """
    Harris Corner。
    1) Ix, Iy
    2) Ix^2, Iy^2, IxIy 做 Gaussian
    3) R = det(M) - k*(trace(M))^2
    4) 閾值 + 以區域最大值篩選
    5) 取前 max_points 個角點 (依照 R 大小)
    回傳： [(y, x, R), ...]
    """
    h, w, _ = img_bgr.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # 計算 Ix, Iy (使用簡單 kernel or Sobel)
    # 與示例相同
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
                # 區域最大值判斷
                local_patch = R[i - 1:i + 2, j - 1:j + 2]
                if R[i, j] == np.max(local_patch):
                    corner_candidates.append((i, j, R[i, j]))

    # 按 R值排序，取前 max_points
    corner_candidates.sort(key=lambda x: x[2], reverse=True)
    corner_candidates = corner_candidates[:max_points]
    return corner_candidates, Ix, Iy

def compute_keypoints_and_descriptors_harris(img_bgr, max_points=200):
    """
    1) HarrisCorner(img_bgr)
    2) 為每個角點產生 descriptor
    回傳:
       kps   : list of (x, y)
       descs : numpy array (N x 128), 其中 N = len(kps)
    """
    corner_candidates, Ix, Iy = HarrisCorner(img_bgr, max_points=max_points)

    # 計算梯度大小 m, 角度 theta
    m, theta = calc_orientation(Ix, Iy)

    kps = []
    descs = []
    h, w, _ = img_bgr.shape
    # 為了安全，避免邊界(要能取到 16x16 patch)
    margin = 8
    for (yy, xx, val) in corner_candidates:
        if yy < margin or yy >= (h - margin):
            continue
        if xx < margin or xx >= (w - margin):
            continue
        kps.append((xx, yy))  # OpenCV風格是 (x, y)
        descriptor = gen_descriptor(yy, xx, m, theta)
        descs.append(descriptor)
    descs = np.array(descs, dtype=np.float32)
    return kps, descs

#############################
# (B) RANSAC for 平移 (dx, dy)
#############################
def simple_match(kpsA, descA, kpsB, descB, desc_thresh=1.0):
    """
    簡單最近鄰匹配:
    desc_thresh 為「最小的距離」若大於此值就丟棄(此處用 L2)。
    回傳:
       matches: [ ((xA,yA), (xB,yB)), ... ]
    """
    matches = []
    for i in range(len(descA)):
        best_dist = float('inf')
        best_idx = -1
        vecA = descA[i]
        for j in range(len(descB)):
            vecB = descB[j]
            diff = vecA - vecB
            dist = np.dot(diff, diff)  # L2距離平方
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        if best_dist < desc_thresh:
            matches.append((kpsA[i], kpsB[best_idx]))
    return matches

def ransac(matches, dist_sq_thresh=3):
    """
    用平移 (dx, dy) 做投票式 RANSAC
    matches: list of ((xA,yA), (xB,yB))
    回傳:
      best_move (dx, dy)
      best_pair
    """
    if len(matches) == 0:
        return (0, 0), None
    move_candidates = []
    for (ptA, ptB) in matches:
        dx = ptA[0] - ptB[0]
        dy = ptA[1] - ptB[1]
        move_candidates.append((dx, dy))

    best_score = -1
    best_move = (0, 0)
    best_pair = None
    for i, candi in enumerate(move_candidates):
        dx_ref, dy_ref = candi
        # 計算所有 move_candidates 與 candi 的差值
        deltas = [(mc[0] - dx_ref, mc[1] - dy_ref) for mc in move_candidates]
        dist_sq_list = [d[0]**2 + d[1]**2 for d in deltas]
        vote_count = np.count_nonzero(np.array(dist_sq_list) < dist_sq_thresh)
        if vote_count > best_score:
            best_score = vote_count
            best_move = candi
            best_pair = matches[i]
    return best_move, best_pair

def compute_shift_harris(imgA, imgB, ransac_thr=3, desc_thresh=1.0):
    """
    以 HarrisCorner + 簡易描述子 + 簡單匹配 + RANSAC計算平移量
    """
    kpsA, descA = compute_keypoints_and_descriptors_harris(imgA, max_points=200)
    kpsB, descB = compute_keypoints_and_descriptors_harris(imgB, max_points=200)

    # 最近鄰匹配
    matches = simple_match(kpsA, descA, kpsB, descB, desc_thresh=desc_thresh)

    # RANSAC
    best_move, best_pair = ransac(matches, dist_sq_thresh=ransac_thr)
    return best_move, best_pair

#############################
# (C) 投影、padding、blend
#############################
def cylindrical_projection(img_bgr, focal_len):
    """
    將影像做圓柱投影
    """
    h, w = img_bgr.shape[:2]
    center_y = h // 2
    center_x = w // 2
    output = np.zeros_like(img_bgr, dtype=np.uint8)

    for yy in range(h):
        for xx in range(w):
            x_dist = xx - center_x
            y_dist = yy - center_y
            x_mapped = round(focal_len * math.atan(x_dist / focal_len)) + center_x
            denom = math.sqrt(x_dist**2 + focal_len**2)
            y_mapped = round(focal_len * (y_dist / denom)) + center_y

            if 0 <= x_mapped < w and 0 <= y_mapped < h:
                output[y_mapped, x_mapped] = img_bgr[yy, xx]
    return output

def pad_image(img_bgr, move_x, move_y):
    """
    依 move_x, move_y 對影像做 zero padding 達到平移目的
    """
    move_x = int(round(move_x))
    move_y = int(round(move_y))
    if move_x >= 0 and move_y >= 0:
        padded = np.pad(img_bgr, ((move_y, 0), (move_x, 0), (0, 0)), 'constant')
    elif move_x >= 0 and move_y < 0:
        padded = np.pad(img_bgr, ((0, -move_y), (move_x, 0), (0, 0)), 'constant')
    elif move_x < 0 and move_y >= 0:
        padded = np.pad(img_bgr, ((move_y, 0), (0, -move_x), (0, 0)), 'constant')
    else:
        padded = np.pad(img_bgr, ((0, -move_y), (0, -move_x), (0, 0)), 'constant')
    return padded

def blend_two_images(shift_vec, ref_match, imgA, imgB):
    """
    使用 shift_vec(dx, dy) 與簡易線性融合將 imgB 拼接到 imgA
    """
    dx, dy = shift_vec

    # 若 dx < 0，交換順序
    if dx < 0:
        dx = -dx
        dy = -dy
        ref_match = (ref_match[1], ref_match[0])
        imgA, imgB = imgB, imgA

    # 計算 x方向 padding
    padA_x = imgB.shape[1] - imgA.shape[1] + ref_match[0][0] - ref_match[1][0]
    padB_x = ref_match[0][0] - ref_match[1][0]
    overlap_range = ref_match[1][0] - ref_match[0][0] + imgA.shape[1]

    shiftA = pad_image(imgA, -padA_x, -dy)
    shiftB = pad_image(imgB, padB_x, dy)

    HH = max(shiftA.shape[0], shiftB.shape[0])
    WW = max(shiftA.shape[1], shiftB.shape[1])
    canvasA = np.zeros((HH, WW, 3), dtype=np.float32)
    canvasB = np.zeros((HH, WW, 3), dtype=np.float32)

    canvasA[:shiftA.shape[0], :shiftA.shape[1]] = shiftA
    canvasB[:shiftB.shape[0], :shiftB.shape[1]] = shiftB

    result = np.zeros((HH, WW, 3), dtype=np.float32)
    overlap_counter = 0

    for cc in range(WW):
        colA = canvasA[:, cc, :]
        colB = canvasB[:, cc, :]
        countA = np.count_nonzero(colA)
        countB = np.count_nonzero(colB)

        if (countA > 0) and (countB > 0):
            alpha = overlap_counter / overlap_range if overlap_range != 0 else 0
            overlap_counter += 1
            result[:, cc, :] = (1 - alpha) * colA + alpha * colB
        elif (countA > 0):
            result[:, cc, :] = colA
        elif (countB > 0):
            result[:, cc, :] = colB
        else:
            pass

    return result.astype(np.uint8)

#############################
# (D) Rectangling (裁切全黑區域)
#############################
def rectangle_crop(img, black_threshold=0, extra_margin=15):
    """
    將輸入影像 (BGR) 中灰階大於 black_threshold 的像素視為有效區域，
    找到其最小外框並裁切，若全圖皆低於該值(幾乎全黑)就原圖返回。
    可另外指定 extra_margin，在已確定的外框內再多切除 extra_margin 像素。

    :param img: 輸入彩色影像 (H, W, 3)
    :param black_threshold: int，若 gray <= black_threshold 視為「黑」
    :param extra_margin:    int，再多切除的邊緣像素(往內)

    :return: 裁切後之影像
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) 根據 threshold 做 mask：> threshold 才算有效
    mask = (gray > black_threshold)

    # 2) 找出 mask 為 True 的像素座標
    coords = np.where(mask)
    if coords[0].size == 0:
        # 全圖都小於等於 threshold，視為全黑，直接回傳原圖
        return img

    # 3) 取得最小外框
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # 4) 額外留白邊界(可往內切)
    y_min = max(0, y_min + extra_margin)
    y_max = min(h - 1, y_max - extra_margin)
    x_min = max(0, x_min + extra_margin)
    x_max = min(w - 1, x_max - extra_margin)

    if y_min > y_max or x_min > x_max:
        # 調整後的範圍無效，直接回傳原圖或空影像
        return img

    # 5) 裁切
    return img[y_min:y_max+1, x_min:x_max+1]

#############################
# E) 主程式
#############################
def run_panorama():
    """
    主程式：互動式要求用戶輸入資料夾與 pano.txt，然後執行全景拼接
    使用 Harris Corner + 簡易描述子進行特徵偵測，再根據第一張 & 最後一張 y差做 drift 修正後拼接
    """
    folder_path = input("請輸入圖片資料夾位置 (預設為 .) ：").strip()
    if folder_path == '':
        folder_path = '.'
    if not (folder_path.endswith('/') or folder_path.endswith('\\')):
        folder_path += '/'

    pano_file = input("請輸入 pano.txt 檔案路徑 (若同資料夾僅輸入檔名)：").strip()
    if pano_file == '':
        pano_file = folder_path + "pano.txt"

    # 讀取 pano.txt
    img_paths, focals = read_pano_data(pano_file)
    if len(img_paths) == 0:
        print("在 pano.txt 中找不到任何有效條目，請檢查格式。")
        return
    print("已從 pano.txt 讀取 %d 張影像路徑及其焦距。" % len(img_paths))

    start = time.time()
    images = []

    # 讀取所有影像 & 做圓柱投影
    for p in img_paths:
        full_p = p if os.path.exists(p) else os.path.join(folder_path, os.path.basename(p))
        img = cv2.imread(full_p)
        if img is None:
            print(f"無法讀取：{full_p}")
            images.append(None)
            continue
        images.append(img)

    # 先把每張做圓柱投影
    cyl_imgs = []
    for i, (img, f) in enumerate(zip(images, focals)):
        if img is None:
            cyl_imgs.append(None)
            continue
        cyl = cylindrical_projection(img, f)
        cyl_imgs.append(cyl)
    print("圓柱投影完成，總共 %d 張影像。" % len(cyl_imgs))

    # 第一次迴圈：計算相鄰影像的 shift，但不直接拼接
    shift_list = []
    matched_pairs = []

    second = time.time()
    print("Timer: %.2f 秒 讀取影像、圓柱投影" % (second - start))

    # 計算 pairwise shift
    for i in range(len(cyl_imgs) - 1):
        if cyl_imgs[i] is None or cyl_imgs[i + 1] is None:
            shift_list.append((0, 0))
            matched_pairs.append(((0, 0), (0, 0)))
            continue

        # 若高度不同，先padding
        diff_y = cyl_imgs[i].shape[0] - cyl_imgs[i + 1].shape[0]
        if diff_y != 0:
            cyl_imgs[i + 1] = pad_image(cyl_imgs[i + 1], 0, diff_y)

        print("拼接中：第 %d / %d 張..." % (i + 1, len(cyl_imgs) - 1))
        shift_xy, matched_pair = compute_shift_harris(
            cyl_imgs[i],
            cyl_imgs[i + 1],
            ransac_thr=3,
            desc_thresh=1.0
        )
        shift_list.append(shift_xy)
        matched_pairs.append(matched_pair)

    third = time.time()
    print("Timer: %.2f 秒 Harris角點 + RANSAC 完成" % (third - second))

    # (E) End-to-end alignment (drift 修正)
    # 計算累計移動
    acc_shifts = [(0, 0)]
    for i in range(len(shift_list)):
        px, py = acc_shifts[i]
        dx, dy = shift_list[i]
        acc_shifts.append((px + dx, py + dy))

    final_dx, final_dy = acc_shifts[-1]
    N = len(images)
    if N > 1:
        average_drift = final_dy / (N - 1)
    else:
        average_drift = 0

    new_shift_list = []
    for i, (dx, dy) in enumerate(shift_list):
        dy_new = dy - average_drift
        new_shift_list.append((dx, dy_new))

    # ============= 第二次迴圈：真正拼接 =============
    mosaic = cyl_imgs[0].copy() if cyl_imgs[0] is not None else None
    for i in range(1, N):
        if cyl_imgs[i] is None or mosaic is None:
            continue
        diff_y = mosaic.shape[0] - cyl_imgs[i].shape[0]
        if diff_y != 0:
            cyl_imgs[i] = pad_image(cyl_imgs[i], 0, diff_y)

        shift_xy = new_shift_list[i - 1]
        pair = matched_pairs[i - 1]
        mosaic = blend_two_images(shift_xy, pair, mosaic, cyl_imgs[i])
        print("實際拼接：第 %d / %d 張..." % (i, N - 1))

    if mosaic is None:
        print("無法拼接任何圖片，請檢查資料或參數設定。")
        return

    # 最後裁切
    result_img = rectangle_crop(mosaic)
    save_path = os.path.join(folder_path, "panoroma_harris.jpg")
    cv2.imwrite(save_path, result_img)
    print(f"全景拼接完成，輸出：{save_path}")

    end = time.time()
    print("總共花費 %.2f 秒" % (end - start))

if __name__ == "__main__":
    run_panorama()
