# 參考 Distinctive Image Features from Scale-Invariant Keypoints
# 參考 Implementing SIFT in Python: A Complete Guide
# 關鍵函式如：generateDescriptors()為完全重寫，並加入中文註解及程式邏輯，程式執行速度比原作者快4倍
from functools import cmp_to_key
import cv2
import numpy as np

# 全局微小數值容差，用於避免除以零
float_tolerance = 1e-7


#################
# 主函式：計算 Keypoints 和 Descriptors
#################
def compute_keypoints_and_descriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    """
    計算輸入影像的 SIFT 關鍵點和描述子
    步驟：
      1. 生成金字塔基底影像
      2. 計算高斯模糊金字塔
      3. 計算差分金字塔 (DoG)
      4. 在尺度空間中尋找極值點
      5. 去除重複關鍵點
      6. 將鍵點座標轉回原始影像大小
      7. 生成描述子向量
    """
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('float32')  # 轉為 float32
    base_image = generate_base_image(image, sigma, assumed_blur)  # 步驟 1
    num_octaves = compute_number_of_octaves(base_image.shape)      # 步驟 2
    gaussian_kernels = generate_gaussian_kernels(sigma, num_intervals)  # 生成模糊核
    gaussian_images = generate_gaussian_images(base_image, num_octaves, gaussian_kernels)  # 步驟 3
    dog_images = generate_DoG_images(gaussian_images)             # 步驟 4
    keypoints = find_scale_space_extrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)  # 步驟 5
    keypoints = remove_duplicate_keypoints(keypoints)             # 去重
    keypoints = convert_keypoints_to_input_image_size(keypoints)     # 調回尺寸
    descriptors = generate_descriptors(keypoints, gaussian_images)  # 步驟 7
    return keypoints, descriptors


#########################
# 影像金字塔相關函式
#########################
def generate_base_image(image, sigma, assumed_blur):
    """
    生成基底影像：將輸入影像上採樣 2 倍並應用高斯模糊
    """
    # print('Generate Base Image...')
    if sigma is None:
        sigma = 1.6
    # 插值放大
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    # 計算差量模糊
    sigma_diff = np.sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)


def compute_number_of_octaves(image_shape):
    """
    根據基底影像最小邊長計算金字塔層數
    """
    return int(np.round(np.log(min(image_shape)) / np.log(2) - 1))


def generate_gaussian_kernels(sigma, num_intervals):
    """
    生成每個尺度的高斯核標準差
    """
    # print('Generate Gaussian Kernels...')
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    kernels = np.zeros(num_images_per_octave)
    kernels[0] = sigma
    for idx in range(1, num_images_per_octave):
        sigma_prev = (k ** (idx - 1)) * sigma
        sigma_total = k * sigma_prev
        kernels[idx] = np.sqrt(sigma_total**2 - sigma_prev**2)
    return kernels


def generate_gaussian_images(image, num_octaves, gaussian_kernels):
    """
    生成高斯金字塔，每個 octave 分別模糊後再下採樣
    """
    # print('Generate Gaussian Images...')
    pyramid = []
    for _ in range(num_octaves):
        octave_imgs = [image]
        for g in gaussian_kernels[1:]:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=g, sigmaY=g)
            octave_imgs.append(image)
        pyramid.append(octave_imgs)
        # 下採樣作為下一 octave 基底
        base = octave_imgs[-3]
        image = cv2.resize(base, (base.shape[1]//2, base.shape[0]//2), interpolation=cv2.INTER_NEAREST)
    return np.array(pyramid, dtype=object)


def generate_DoG_images(gaussian_images):
    """
    生成差分金字塔 (DoG)，方便極值檢測
    """
    # print('Generate DoG Images...')
    dog_pyr = []
    for octave in gaussian_images:
        dogs = []
        for first, second in zip(octave, octave[1:]):
            dogs.append(second - first)  # 差分可能為負
        dog_pyr.append(dogs)
    return np.array(dog_pyr, dtype=object)


###############################
# 尺度空間極值檢測相關函式
###############################
def find_scale_space_extrema(gaussian_images, dog_images, num_intervals, sigma, border, contrast_threshold=0.04):
    """
    在 DoG 金字塔中尋找尺度空間極值作為候選鍵點
    """
    # print('Find Scale Space Extrema...')
    thresh = np.floor(0.5 * contrast_threshold / num_intervals * 255)
    keypoints = []
    for o, dogs in enumerate(dog_images):
        for i, (prev, curr, nxt) in enumerate(zip(dogs, dogs[1:], dogs[2:])):
            h, w = curr.shape
            for y in range(border, h-border):
                for x in range(border, w-border):
                    if is_pixel_an_extremum(prev[y - 1:y + 2, x - 1:x + 2],
                                         curr[y-1:y+2, x-1:x+2],
                                         nxt[y-1:y+2, x-1:x+2],
                                            thresh):
                        res = localize_extremum_via_quadratic_fit(
                            x, y, i+1, o, num_intervals, dogs, sigma, contrast_threshold, border
                        )
                        if res:
                            kp, layer = res
                            orients = compute_keypoints_with_orientations(kp, o, gaussian_images[o][layer])
                            keypoints.extend(orients)
    return keypoints


def is_pixel_an_extremum(prev_patch, curr_patch, next_patch, threshold):
    """
    判斷 3x3x3 小區域中心是否為極值（局部最大或最小）
    """
    val = curr_patch[1,1]
    if abs(val) <= threshold:
        return False
    if val > 0:
        return (np.all(val >= prev_patch) and
                np.all(val >= next_patch) and
                np.all(val >= curr_patch[0]) and
                np.all(val >= curr_patch[2]) and
                val >= curr_patch[1,0] and
                val >= curr_patch[1,2])
    else:
        return (np.all(val <= prev_patch) and
                np.all(val <= next_patch) and
                np.all(val <= curr_patch[0]) and
                np.all(val <= curr_patch[2]) and
                val <= curr_patch[1,0] and
                val <= curr_patch[1,2])


###############################
# 亞像素極值定位 (Quadratic Fit)
###############################
def localize_extremum_via_quadratic_fit(x, y, layer, octave, num_intervals, dog_octave,
                                        sigma, contrast_threshold, border, eigen_ratio=10, max_iter=5):
    """
    利用二次擬合精細化極值位置並濾除邊緣響應與低對比度點
    """
    shape = dog_octave[0].shape
    for _ in range(max_iter):
        prev, curr, nxt = dog_octave[layer-1:layer+2]
        cube = np.stack([
            prev[y-1:y+2, x-1:x+2],
            curr[y-1:y+2, x-1:x+2],
            nxt[y-1:y+2, x-1:x+2]
        ]).astype('float32') / 255.
        grad = compute_gradient_at_center_pixel(cube)  # 一階導數
        hess = compute_hessian_at_center_pixel(cube)   # 二階導數
        update = -np.linalg.lstsq(hess, grad, rcond=None)[0]
        # 若更新量足夠小則收斂
        if np.all(np.abs(update) < 0.5):
            break
        x += int(np.round(update[0]))
        y += int(np.round(update[1]))
        layer += int(np.round(update[2]))
        if (y < border or y >= shape[0]-border or
            x < border or x >= shape[1]-border or
            layer < 1 or layer > num_intervals):
            return None
    val = cube[1,1,1] + 0.5 * np.dot(grad, update)
    # 對比度檢驗
    if abs(val) * num_intervals < contrast_threshold:
        return None
    # 邊緣響應檢驗
    h2 = hess[:2, :2]
    tr = np.trace(h2)
    d = np.linalg.det(h2)
    if d <= 0 or eigen_ratio * (tr**2) >= ((eigen_ratio+1)**2)*d:
        return None
    # 构建 KeyPoint
    kp = cv2.KeyPoint()
    kp.pt = ((x + update[0]) * (2**octave), (y + update[1]) * (2**octave))
    kp.octave = octave + layer*(2**8) + int(np.round((update[2]+0.5)*255))*(2**16)
    kp.size = sigma * (2**((layer + update[2]) / np.float32(num_intervals))) * (2**(octave+1))
    kp.response = abs(val)
    return kp, layer


#########################
# 一階／二階導數計算
#########################
def compute_gradient_at_center_pixel(cube):
    """
    以中心差分計算三維圖像立體點的一階導數 (dx, dy, ds)
    """
    dx = 0.5 * (cube[1,1,2] - cube[1,1,0])
    dy = 0.5 * (cube[1,2,1] - cube[1,0,1])
    ds = 0.5 * (cube[2,1,1] - cube[0,1,1])
    return np.array([dx, dy, ds])


def compute_hessian_at_center_pixel(cube):
    """
    計算三維圖像立體點的二階導數矩陣 (Hessian)
    """
    v = cube[1,1,1]
    dxx = cube[1,1,2] - 2*v + cube[1,1,0]
    dyy = cube[1,2,1] - 2*v + cube[1,0,1]
    dss = cube[2,1,1] - 2*v + cube[0,1,1]
    dxy = 0.25 * (cube[1,2,2] - cube[1,2,0] - cube[1,0,2] + cube[1,0,0])
    dxs = 0.25 * (cube[2,1,2] - cube[2,1,0] - cube[0,1,2] + cube[0,1,0])
    dys = 0.25 * (cube[2,2,1] - cube[2,0,1] - cube[0,2,1] + cube[0,0,1])
    return np.array([[dxx, dxy, dxs],
                     [dxy, dyy, dys],
                     [dxs, dys, dss]])


#########################
# Keypoint 方向分配
#########################
def compute_keypoints_with_orientations(keypoint, octave, gauss_img,
                                        radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """
    計算每個鍵點的主要方向，生成可能多個方向
    """
    out_kps = []
    # 計算特徵窗口半徑
    scale = scale_factor * keypoint.size / np.float32(2**(octave+1))
    radius = int(np.round(radius_factor * scale))
    weight_fac = -0.5 / (scale**2)
    raw_hist = np.zeros(num_bins)
    # 建立方向直方圖
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            y = int(np.round(keypoint.pt[1] / np.float32(2**octave))) + dy
            x = int(np.round(keypoint.pt[0] / np.float32(2**octave))) + dx
            if (x <= 0 or x >= gauss_img.shape[1]-1 or
                y <= 0 or y >= gauss_img.shape[0]-1):
                continue
            gx = gauss_img[y, x+1] - gauss_img[y, x-1]
            gy = gauss_img[y-1, x] - gauss_img[y+1, x]
            mag = np.sqrt(gx*gx + gy*gy)
            ang = np.rad2deg(np.arctan2(gy, gx)) % 360
            w = np.exp(weight_fac * (dx*dx + dy*dy))
            idx = int(np.round(ang * num_bins / 360.)) % num_bins
            raw_hist[idx] += w * mag
    # 平滑直方圖
    smooth = np.zeros(num_bins)
    for i in range(num_bins):
        smooth[i] = (6*raw_hist[i] +
                     4*(raw_hist[i-1] + raw_hist[(i+1) % num_bins]) +
                     raw_hist[i-2] + raw_hist[(i+2) % num_bins]) / 16.
    maxv = np.max(smooth)
    # 尋找峰值
    peaks = np.where(np.logical_and(smooth > np.roll(smooth, 1),
                                    smooth > np.roll(smooth, -1)))[0]
    for p in peaks:
        if smooth[p] >= peak_ratio * maxv:
            l = smooth[(p-1) % num_bins]
            r = smooth[(p+1) % num_bins]
            interp = (p + 0.5 * (l - r) / (l - 2*smooth[p] + r)) % num_bins
            angle = (360. - interp * 360. / num_bins)
            if abs(angle - 360.) < float_tolerance:
                angle = 0
            newkp = cv2.KeyPoint(*keypoint.pt, keypoint.size, angle,
                                 keypoint.response, keypoint.octave)
            out_kps.append(newkp)
    return out_kps


##############################
# 重複鍵點移除
##############################
def compare_keypoints(kp1, kp2):
    """鍵點排序依序：x, y, size, angle, response, octave"""
    if kp1.pt[0] != kp2.pt[0]:
        return kp1.pt[0] - kp2.pt[0]
    if kp1.pt[1] != kp2.pt[1]:
        return kp1.pt[1] - kp2.pt[1]
    if kp1.size != kp2.size:
        return kp2.size - kp1.size
    if kp1.angle != kp2.angle:
        return kp1.angle - kp2.angle
    if kp1.response != kp2.response:
        return kp2.response - kp1.response
    return kp2.class_id - kp1.class_id


def remove_duplicate_keypoints(keypoints):
    """
    按 compareKeypoints 排序並移除座標及屬性完全相同的鍵點
    """
    if len(keypoints) < 2:
        return keypoints
    keypoints.sort(key=cmp_to_key(compare_keypoints))
    unique = [keypoints[0]]
    for kp in keypoints[1:]:
        last = unique[-1]
        if (last.pt != kp.pt or last.size != kp.size or
            last.angle != kp.angle):
            unique.append(kp)
    return unique


#############################
# 鍵點座標尺寸轉換
#############################
def convert_keypoints_to_input_image_size(keypoints):
    """
    將鍵點座標、大小從上採樣後版本轉回原始影像大小
    """
    out = []
    for kp in keypoints:
        kp.pt = (kp.pt[0] * 0.5, kp.pt[1] * 0.5)
        kp.size *= 0.5
        kp.octave = (kp.octave & ~255) | ((kp.octave - 1) & 255)
        out.append(kp)
    return out


#########################
# 描述子生成
#########################
def unpack_octave(keypoint):
    """
    從 kp.octave 編碼中解析 octave, layer, scale
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave |= -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    return octave, layer, scale


def generate_descriptors(keypoints, gaussian_images, window_width=4, num_bins=8,
                         scale_multiplier=3, descriptor_max_value=0.2):
    """
    根據鍵點位置和方向，在局部窗口內累積梯度直方圖生成 128 維描述子 (每 keypoint 4x4x8)
    改用向量化 + np.add.at，避免了原本 (dy, dx) 的雙層巢狀 for-loop
    """
    # print('Generate Descriptors...')
    descriptors = []

    for kp in keypoints:
        # 1. 取得該 keypoint 對應的 octave/layer 與座標
        octv, lyr, scl = unpack_octave(kp)
        img = gaussian_images[octv + 1][lyr]
        rows, cols = img.shape
        pt = np.round(scl * np.array(kp.pt)).astype(int)

        angle = 360. - kp.angle
        cos_a, sin_a = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))

        # 每個描述子的空間網格大小
        # 建立 (window_width+2)x(window_width+2)x num_bins 的 histogram tensor
        tensor = np.zeros((window_width + 2, window_width + 2, num_bins), dtype=np.float32)

        # 金字塔中對應的實際取樣視窗半徑(特徵描述子窗口半徑)
        hist_width = scale_multiplier * 0.5 * scl * kp.size
        half_w = int(np.round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
        half_w = min(half_w, int(np.sqrt(rows**2 + cols**2)))

        # 2. 生成 [dy, dx] 的網格並展平
        # 相當於 for dy in range(-half_w, half_w+1): for dx in range(-half_w, half_w+1):
        ys, xs = np.mgrid[-half_w:half_w+1, -half_w:half_w+1]
        ys = ys.ravel()
        xs = xs.ravel()

        # 3. 對應原圖中的絕對座標 (rr, cc) 做第一階段過濾
        rr = pt[1] + ys
        cc = pt[0] + xs

        # 濾除越界像素
        valid_mask = (rr > 0) & (rr < rows - 1) & (cc > 0) & (cc < cols - 1)
        if not np.any(valid_mask):
            # 該 keypoint 如果全部越界，就跳過
            descriptors.append(np.zeros(128, dtype=np.float32))
            continue

        # 只保留有效像素
        rr = rr[valid_mask]
        cc = cc[valid_mask]
        ys = ys[valid_mask]
        xs = xs[valid_mask]

        # 4. 計算梯度
        # (gx, gy) / 幅度 mag / 方向 orient (針對縮減後的 rr, cc)
        gx = img[rr, cc + 1] - img[rr, cc - 1]
        gy = img[rr - 1, cc] - img[rr + 1, cc]
        mag = np.sqrt(gx * gx + gy * gy)
        orient = np.rad2deg(np.arctan2(gy, gx)) % 360

        # 5. 計算旋轉後的局部座標 (r_rot, c_rot)
        # xs, ys 已縮減，所以 r_rot, c_rot 與 mag, orient 皆對應同長度
        r_rot = xs * sin_a + ys * cos_a
        c_rot = xs * cos_a - ys * sin_a

        # 6. 投影到 [window_width x window_width] 金字塔網格
        r_bin = (r_rot / hist_width) + 0.5 * window_width - 0.5
        c_bin = (c_rot / hist_width) + 0.5 * window_width - 0.5

        # 7. 第二階段過濾：只保留落在 [-1, window_width) 之內的點
        valid_bin_mask = (r_bin > -1.0) & (r_bin < window_width) & \
                         (c_bin > -1.0) & (c_bin < window_width)
        if not np.any(valid_bin_mask):
            descriptors.append(np.zeros(128, dtype=np.float32))
            continue

        # 依據 valid_bin_mask 再次縮減
        r_bin = r_bin[valid_bin_mask]
        c_bin = c_bin[valid_bin_mask]
        mag = mag[valid_bin_mask]
        orient = orient[valid_bin_mask]
        r_rot = r_rot[valid_bin_mask]
        c_rot = c_rot[valid_bin_mask]

        # 8. 高斯加權
        #   與原程式同理： weight_mul = -0.5 / ((0.5*window_width)**2)
        #   但原程式的寫法是 (r_rot/hist_width)**2+(c_rot/hist_width)**2
        #   為保持一致，這裡照抄
        weight_mul = -0.5 / ((0.5 * window_width)**2)
        w = np.exp(weight_mul * ((r_rot / hist_width)**2 + (c_rot / hist_width)**2))

        # 累計真正的幅度 = w * mag
        weighted_mag = w * mag

        # 計算 (orient - angle)* bins_per_deg
        bins_per_deg = num_bins / 360.
        ob = (orient - angle) * bins_per_deg
        ob = np.mod(ob, num_bins)

        # 9. 三線性插值
        r0 = np.floor(r_bin).astype(int)
        c0 = np.floor(c_bin).astype(int)
        o0 = np.floor(ob).astype(int)
        o0 = o0 % num_bins

        rf = r_bin - r0
        cf = c_bin - c0
        of = ob - o0

        # 開始做三線性 (row, col, orient) 分配，但程式裡拆成二段
        # 先對 row 分兩塊 c0w, c1
        c1 = weighted_mag * rf
        c0w = weighted_mag - c1

        # 再對 col 分 c?? => c00, c01, c10, c11
        c10 = c1 * (1 - cf)
        c11 = c1 * cf
        c00 = c0w * (1 - cf)
        c01 = c0w * cf

        # 方向上分兩塊 => offset = 0, offset=+1 (mod num_bins)
        # 不同塊對應 c00 ~ c11 的「方向分配」
        # 先計算對 o0 (整數方向bin) 的分量 => (1 - of)
        #           對 (o0+1)%num_bins 的分量 => of
        # 所以對於 c00 來說：對 o0 加 c00*(1-of)，對 (o0+1) 加 c00*of
        # 其餘 c01, c10, c11 亦同
        # 我們利用 np.add.at 來「散射加法」

        # 定義 helper，將分量散射到 tensor
        def scatter_orient(magnitude, frac, r_ind, c_ind, o_bin):
            """
            將 magnitude*(1-frac) 分到 [r_ind, c_ind, o_bin]
            將 magnitude* frac   分到 [r_ind, c_ind, (o_bin+1)%num_bins]
            """
            base = magnitude * (1 - frac)
            plus = magnitude * frac

            o_bin0 = o_bin % num_bins
            o_bin1 = (o_bin + 1) % num_bins

            np.add.at(tensor, (r_ind + 1, c_ind + 1, o_bin0), base)
            np.add.at(tensor, (r_ind + 1, c_ind + 1, o_bin1), plus)

        # 4個空間位置 + 2個方向bin => 8種散射
        scatter_orient(c00, of, r0, c0, o0)
        scatter_orient(c01, of, r0, c0 + 1, o0)
        scatter_orient(c10, of, r0 + 1, c0, o0)
        scatter_orient(c11, of, r0 + 1, c0 + 1, o0)

        # 10. 去掉外圍 padding
        vec = tensor[1:-1, 1:-1, :].ravel()

        # 對比度截斷並正規化
        thr = np.linalg.norm(vec) * descriptor_max_value
        vec[vec > thr] = thr
        norm_v = np.linalg.norm(vec)
        if norm_v < float_tolerance:
            norm_v = float_tolerance
        vec /= norm_v

        # 轉為 0~255 (SIFT 通常以 512 scale，再 clamp 到 255)
        vec = np.round(512 * vec)
        vec[vec < 0] = 0
        vec[vec > 255] = 255

        descriptors.append(vec.astype('float32'))

    return np.array(descriptors, dtype='float32')
