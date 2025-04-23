import os
import cv2
import numpy as np
import math
import time
from sift_impl import compute_keypoints_and_descriptors


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
# 1) SIFT + RANSAC (feature detection and matching)
#############################
def compute_shift_sift(imgA, imgB, ransac_thr=3, desc_thresh=25000):
    """
    使用自製 SIFT 進行 Keypoints + Descriptors，
    再用簡易最近鄰配對 + RANSAC 找 (dx,dy).
    :param desc_thresh: 用來判斷特徵向量距離（L2距離）的閾值
    """
    # 1) 計算 SIFT Keypoints & Descriptors
    kpsA, descA = compute_keypoints_and_descriptors(imgA)
    kpsB, descB = compute_keypoints_and_descriptors(imgB)

    # 2) 做最近鄰匹配
    matches = []
    for i in range(len(descA)):
        best_dist = float('inf')
        best_idx = -1
        for j in range(len(descB)):
            # SIFT desc => 128維
            d = descA[i] - descB[j]
            dist = np.dot(d,d)  # L2距離平方
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        if best_dist < desc_thresh and best_idx != -1:
            # 建立 ( (xA,yA), (xB,yB) ) 形式
            (xA, yA) = kpsA[i].pt
            (xB, yB) = kpsB[best_idx].pt
            # 轉成整數或保留 float 也可
            matches.append(((xA, yA), (xB, yB)))

    # 3) RANSAC => best shift
    best_move, best_pair = ransac(matches, dist_sq_thresh=ransac_thr)
    return best_move, best_pair


def ransac(matches, dist_sq_thresh=3):
    """
    用平移 (dx, dy) 做投票式 RANSAC
    """
    if len(matches) == 0:
        return (0, 0), None
    move_candidates = []
    for (a, b) in matches:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        move_candidates.append((dx, dy))

    best_score = -1
    best_move = (0, 0)
    best_pair = None

    for i, candi in enumerate(move_candidates):
        dx_ref, dy_ref = candi
        deltas = [(mc[0] - dx_ref, mc[1] - dy_ref) for mc in move_candidates]
        dist_sq_list = [d[0] ** 2 + d[1] ** 2 for d in deltas]
        vote_count = np.count_nonzero(np.array(dist_sq_list) < dist_sq_thresh)
        if vote_count > best_score:
            best_score = vote_count
            best_move = candi
            best_pair = matches[i]
    return best_move, best_pair


#############################
# 2) 影像投影、拼接(image matching and blending)
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
    if dx < 0:
        dx = -dx
        dy = -dy
        ref_match = (ref_match[1], ref_match[0])
        imgA, imgB = imgB, imgA

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
# 3) Rectangling
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
# 4) 主程式
#############################
def run_panorama():
    """
    主程式：互動式要求用戶輸入資料夾與 pano.txt，然後執行全景拼接
    使用 SIFT 進行特徵偵測，再根據第一張 & 最後一張 y差做 drift 修正後拼接
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
    # shift_list[i] 表示第 i+1 張 (cyl_imgs[i+1]) 相對第 i 張 (cyl_imgs[i]) 的 (dx, dy)
    shift_list = []
    matched_pairs = []

    mosaic_height_list = [cyl_imgs[0].shape[0]]  # 記錄拼接後實際高度(可選)

    second = time.time()
    print("Timer: %.2f 秒 讀取影像、圓柱投影、計算drift" % (second - start))

    #############################
    # 5) End to end alignment (去除圖片drift高低差問題)
    #############################
    for i in range(len(cyl_imgs) - 1):
        if cyl_imgs[i] is None or cyl_imgs[i + 1] is None:
            shift_list.append((0, 0))
            matched_pairs.append(((0, 0), (0, 0)))
            mosaic_height_list.append(mosaic_height_list[-1])
            continue

        # 若高不同，先做 padding
        diff_y = cyl_imgs[i].shape[0] - cyl_imgs[i + 1].shape[0]
        if diff_y != 0:
            cyl_imgs[i + 1] = pad_image(cyl_imgs[i + 1], 0, diff_y)

        print("拼接中：第 %d / %d 張..." % (i + 1, len(cyl_imgs) - 1))
        shift_xy, matched_pair = compute_shift_sift(cyl_imgs[i], cyl_imgs[i + 1], ransac_thr=3, desc_thresh=25000)
        shift_list.append(shift_xy)
        matched_pairs.append(matched_pair)

        # 這裡可選擇記錄「拼好之後的高度」，但要真的拼才知道。範例就先略過

    third = time.time()
    print("Timer: %.2f 秒 SIFT運算" % (third - second))
    # 計算「累計」移動 y
    # acc_shifts[i] = 第 i 張影像(相對於第 0 張) 的累計 shift
    # 其中 acc_shifts[0] = (0,0)
    acc_shifts = [(0, 0)]
    for i in range(len(shift_list)):
        prev_x, prev_y = acc_shifts[i]
        cur_dx, cur_dy = shift_list[i]
        acc_shifts.append((prev_x + cur_dx, prev_y + cur_dy))

    # 現在 acc_shifts[-1] 就是「最後一張相對第一張」的累計移動
    final_dx, final_dy = acc_shifts[-1]

    # 假設我們只想修正 y 的 drift (高度差)
    # => average_drift = final_dy / (N-1)  (N=len(images))
    N = len(images)
    if N > 1:
        average_drift = final_dy / (N - 1)
    else:
        average_drift = 0

    # 我們將 shift_list[i] 的 dy 依比例扣除 i+1 的 drift
    # 例如第 i 次移動對應「從 i → i+1」的影像
    # 從理論上：acc_shifts[i+1].y = acc_shifts[i].y + shift_list[i].y
    # drift 分配 => shift_list[i].y_new = shift_list[i].y - (i+1)* (average_drift / (N-1))?
    # 但比較簡單：我們直接從acc_shifts計算 ref
    # 這裡示範最直覺的方式：
    new_shift_list = []
    for i, (dx, dy) in enumerate(shift_list):
        # i 代表第 i 次移動 (i=0 => 1st move => 0->1)
        # 調整後：dy_new = dy - (i+1)*average_drift/ ???
        # 其實最簡單：把 "average_drift" 分攤到每一段 => each dy -= average_drift
        dy_new = dy - average_drift
        new_shift_list.append((dx, dy_new))

    # ============= 第二次迴圈：真正拼接 =============
    # 用 new_shift_list 重新 build 全景
    mosaic = cyl_imgs[0].copy()
    for i in range(1, N):
        if cyl_imgs[i] is None:
            continue
        # 先保證高度
        diff_y = mosaic.shape[0] - cyl_imgs[i].shape[0]
        if diff_y != 0:
            cyl_imgs[i] = pad_image(cyl_imgs[i], 0, diff_y)

        shift_xy = new_shift_list[i - 1]
        pair = matched_pairs[i - 1]
        mosaic = blend_two_images(shift_xy, pair, mosaic, cyl_imgs[i])
        print("實際拼接：第 %d / %d 張..." % (i, N - 1))

    result_img = rectangle_crop(mosaic)
    save_path = os.path.join(folder_path, "panoroma.jpg")
    cv2.imwrite(save_path, result_img)
    print(f"全景拼接完成，輸出：{save_path}")
    end = time.time()
    print("總共花費 %.2f 秒" % (end - start))

if __name__ == "__main__":
    run_panorama()
