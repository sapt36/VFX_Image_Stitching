# 參考 Distinctive Image Features from Scale-Invariant Keypoints 論文實作 (https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)


from functools import cmp_to_key
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from numpy import all, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, deg2rad, rad2deg, where, zeros, floor, round, float32
from numpy.linalg import det, lstsq, norm

# 全局微小數值容差，用於避免除以零
float_tolerance = 1e-7

#################
# 主函式：計算 Keypoints 和 Descriptors
#################
def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
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
    image = image.astype('float32')  # 轉為 float32
    base_image = generateBaseImage(image, sigma, assumed_blur)  # 步驟 1
    num_octaves = computeNumberOfOctaves(base_image.shape)  # 步驟 2
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals)  # 生成模糊核
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)  # 步驟 3
    dog_images = generateDoGImages(gaussian_images)  # 步驟 4
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)  # 步驟 5
    keypoints = removeDuplicateKeypoints(keypoints)  # 去重
    keypoints = convertKeypointsToInputImageSize(keypoints)  # 調回尺寸
    descriptors = generateDescriptors(keypoints, gaussian_images)  # 步驟 7
    return keypoints, descriptors

#########################
# 影像金字塔相關函式
#########################
def generateBaseImage(image, sigma, assumed_blur):
    """
    生成基底影像：將輸入影像上採樣 2 倍並應用高斯模糊
    """
    print('Generate Base Image...')
    if sigma is None:
        sigma = 1.6
    # 插值放大
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    # 計算差量模糊
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

def computeNumberOfOctaves(image_shape):
    """
    根據基底影像最小邊長計算金字塔層數
    """
    return int(round(log(min(image_shape)) / log(2) - 1))

def generateGaussianKernels(sigma, num_intervals):
    """
    生成每個尺度的高斯核標準差
    """
    print('Generate Gaussian Kernels...')
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    kernels = zeros(num_images_per_octave)
    kernels[0] = sigma
    for idx in range(1, num_images_per_octave):
        sigma_prev = (k ** (idx - 1)) * sigma
        sigma_total = k * sigma_prev
        kernels[idx] = sqrt(sigma_total**2 - sigma_prev**2)
    return kernels

def generateGaussianImages(image, num_octaves, gaussian_kernels):
    """
    生成高斯金字塔，每個 octave 分別模糊後再下採樣
    """
    print('Generate Gaussian Images...')
    pyramid = []
    for _ in range(num_octaves):
        octave_imgs = [image]
        for g in gaussian_kernels[1:]:
            image = GaussianBlur(image, (0, 0), sigmaX=g, sigmaY=g)
            octave_imgs.append(image)
        pyramid.append(octave_imgs)
        # 下採樣作為下一 octave 基底
        base = octave_imgs[-3]
        image = resize(base, (base.shape[1]//2, base.shape[0]//2), interpolation=INTER_NEAREST)
    return array(pyramid, dtype=object)

def generateDoGImages(gaussian_images):
    """
    生成差分金字塔 (DoG)，方便極值檢測
    """
    print('Generate DoG Images...')
    dog_pyr = []
    for octave in gaussian_images:
        dogs = []
        for first, second in zip(octave, octave[1:]):
            dogs.append(subtract(second, first))  # 差分可能為負
        dog_pyr.append(dogs)
    return array(dog_pyr, dtype=object)

###############################
# 尺度空間極值檢測相關函式
###############################
def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, border, contrast_threshold=0.04):
    """
    在 DoG 金字塔中尋找尺度空間極值作為候選鍵點
    """
    print('Find Scale Space Extrema...')
    thresh = floor(0.5 * contrast_threshold / num_intervals * 255)
    keypoints = []
    for o, dogs in enumerate(dog_images):
        for i, (prev, curr, nxt) in enumerate(zip(dogs, dogs[1:], dogs[2:])):
            h, w = curr.shape
            for y in range(border, h-border):
                for x in range(border, w-border):
                    if isPixelAnExtremum(prev[y-1:y+2, x-1:x+2], curr[y-1:y+2, x-1:x+2], nxt[y-1:y+2, x-1:x+2], thresh):
                        res = localizeExtremumViaQuadraticFit(x, y, i+1, o, num_intervals, dogs, sigma, contrast_threshold, border)
                        if res:
                            kp, layer = res
                            orients = computeKeypointsWithOrientations(kp, o, gaussian_images[o][layer])
                            keypoints.extend(orients)
    return keypoints

def isPixelAnExtremum(prev_patch, curr_patch, next_patch, threshold):
    """
    判斷 3x3x3 小區域中心是否為極值（局部最大或最小）
    """
    val = curr_patch[1,1]
    if abs(val) <= threshold:
        return False
    if val > 0:
        return all(val>=prev_patch) and all(val>=next_patch) and all(val>=curr_patch[0]) and all(val>=curr_patch[2]) and val>=curr_patch[1,0] and val>=curr_patch[1,2]
    else:
        return all(val<=prev_patch) and all(val<=next_patch) and all(val<=curr_patch[0]) and all(val<=curr_patch[2]) and val<=curr_patch[1,0] and val<=curr_patch[1,2]

###############################
# 亞像素極值定位 (Quadratic Fit)
###############################
def localizeExtremumViaQuadraticFit(x, y, layer, octave, num_intervals, dog_octave, sigma, contrast_threshold, border, eigen_ratio=10, max_iter=5):
    """
    利用二次擬合精細化極值位置並濾除邊緣響應與低對比度點
    """
    shape = dog_octave[0].shape
    for _ in range(max_iter):
        prev, curr, nxt = dog_octave[layer-1:layer+2]
        cube = stack([prev[y-1:y+2, x-1:x+2], curr[y-1:y+2, x-1:x+2], nxt[y-1:y+2, x-1:x+2]]).astype('float32')/255.
        grad = computeGradientAtCenterPixel(cube)  # 一階導數
        hess = computeHessianAtCenterPixel(cube)   # 二階導數
        update = -lstsq(hess, grad, rcond=None)[0]
        # 若更新量足夠小則收斂
        if all(abs(update) < 0.5):
            break
        x += int(round(update[0])); y += int(round(update[1])); layer += int(round(update[2]))
        if y<border or y>=shape[0]-border or x<border or x>=shape[1]-border or layer<1 or layer>num_intervals:
            return None
    val = cube[1,1,1] + 0.5*dot(grad, update)
    # 對比度檢驗
    if abs(val)*num_intervals < contrast_threshold:
        return None
    # 邊緣響應檢驗
    h2 = hess[:2,:2]; tr=trace(h2); d=det(h2)
    if d<=0 or eigen_ratio*(tr**2)>=((eigen_ratio+1)**2)*d:
        return None
    # 构建 KeyPoint
    kp = KeyPoint()
    kp.pt = ((x+update[0])*(2**octave), (y+update[1])*(2**octave))
    kp.octave = octave + layer*(2**8) + int(round((update[2]+0.5)*255))*(2**16)
    kp.size = sigma*(2**((layer+update[2])/float32(num_intervals)))*(2**(octave+1))
    kp.response = abs(val)
    return kp, layer

#########################
# 一階／二階導數計算
#########################
def computeGradientAtCenterPixel(cube):
    """
    以中心差分計算三維圖像立體點的一階導數 (dx, dy, ds)
    """
    dx = 0.5*(cube[1,1,2] - cube[1,1,0])
    dy = 0.5*(cube[1,2,1] - cube[1,0,1])
    ds = 0.5*(cube[2,1,1] - cube[0,1,1])
    return array([dx, dy, ds])

def computeHessianAtCenterPixel(cube):
    """
    計算三維圖像立體點的二階導數矩陣 (Hessian)
    """
    v = cube[1,1,1]
    dxx = cube[1,1,2] - 2*v + cube[1,1,0]
    dyy = cube[1,2,1] - 2*v + cube[1,0,1]
    dss = cube[2,1,1] - 2*v + cube[0,1,1]
    dxy = 0.25*(cube[1,2,2]-cube[1,2,0]-cube[1,0,2]+cube[1,0,0])
    dxs = 0.25*(cube[2,1,2]-cube[2,1,0]-cube[0,1,2]+cube[0,1,0])
    dys = 0.25*(cube[2,2,1]-cube[2,0,1]-cube[0,2,1]+cube[0,0,1])
    return array([[dxx,dxy,dxs],[dxy,dyy,dys],[dxs,dys,dss]])

#########################
# Keypoint 方向分配
#########################
def computeKeypointsWithOrientations(keypoint, octave, gauss_img, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """
    計算每個鍵點的主要方向，生成可能多個方向版本
    """
    out_kps = []
    # 計算特徵窗口半徑
    scale = scale_factor*keypoint.size/float32(2**(octave+1))
    radius = int(round(radius_factor*scale))
    weight_fac = -0.5/(scale**2)
    raw_hist = zeros(num_bins)
    # 建立方向直方圖
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            y = int(round(keypoint.pt[1]/float32(2**octave)))+dy
            x = int(round(keypoint.pt[0]/float32(2**octave)))+dx
            if x<=0 or x>=gauss_img.shape[1]-1 or y<=0 or y>=gauss_img.shape[0]-1: continue
            gx = gauss_img[y, x+1]-gauss_img[y, x-1]
            gy = gauss_img[y-1, x]-gauss_img[y+1, x]
            mag = sqrt(gx*gx+gy*gy)
            ang = rad2deg(arctan2(gy, gx)) % 360
            w = exp(weight_fac*(dx*dx+dy*dy))
            idx = int(round(ang*num_bins/360.))%num_bins
            raw_hist[idx]+=w*mag
    # 平滑直方圖
    smooth = zeros(num_bins)
    for i in range(num_bins):
        smooth[i] = (6*raw_hist[i]+4*(raw_hist[i-1]+raw_hist[(i+1)%num_bins])+raw_hist[i-2]+raw_hist[(i+2)%num_bins])/16.
    maxv = max(smooth)
    # 尋找峰值
    peaks = where(logical_and(smooth>roll(smooth,1), smooth>roll(smooth,-1)))[0]
    for p in peaks:
        if smooth[p]>=peak_ratio*maxv:
            l, r = smooth[(p-1)%num_bins], smooth[(p+1)%num_bins]
            interp = (p+0.5*(l-r)/(l-2*smooth[p]+r))%num_bins
            angle = (360.-interp*360./num_bins)
            if abs(angle-360.)<float_tolerance: angle=0
            newkp = KeyPoint(*keypoint.pt, keypoint.size, angle, keypoint.response, keypoint.octave)
            out_kps.append(newkp)
    return out_kps

##############################
# 重複鍵點移除
##############################
def compareKeypoints(kp1, kp2):
    """鍵點排序依序：x, y, size, angle, response, octave"""
    if kp1.pt[0]!=kp2.pt[0]: return kp1.pt[0]-kp2.pt[0]
    if kp1.pt[1]!=kp2.pt[1]: return kp1.pt[1]-kp2.pt[1]
    if kp1.size!=kp2.size: return kp2.size-kp1.size
    if kp1.angle!=kp2.angle: return kp1.angle-kp2.angle
    if kp1.response!=kp2.response: return kp2.response-kp1.response
    return kp2.class_id-kp1.class_id

def removeDuplicateKeypoints(keypoints):
    """
    按 compareKeypoints 排序並移除座標及屬性完全相同的鍵點
    """
    if len(keypoints)<2: return keypoints
    keypoints.sort(key=cmp_to_key(compareKeypoints))
    unique=[keypoints[0]]
    for kp in keypoints[1:]:
        last=unique[-1]
        if last.pt!=kp.pt or last.size!=kp.size or last.angle!=kp.angle:
            unique.append(kp)
    return unique

#############################
# 鍵點座標尺寸轉換
#############################
def convertKeypointsToInputImageSize(keypoints):
    """
    將鍵點座標、大小從上採樣後版本轉回原始影像大小
    """
    out=[]
    for kp in keypoints:
        kp.pt=(kp.pt[0]*0.5, kp.pt[1]*0.5)
        kp.size*=0.5
        kp.octave=(kp.octave & ~255) | ((kp.octave-1)&255)
        out.append(kp)
    return out

#########################
# 描述子生成
#########################
def unpackOctave(keypoint):
    """
    從 kp.octave 編碼中解析 octave, layer, scale
    """
    octave = keypoint.octave & 255
    layer  = (keypoint.octave>>8) & 255
    if octave>=128: octave|=-128
    scale = 1/float32(1<<octave) if octave>=0 else float32(1<<-octave)
    return octave, layer, scale

def generateDescriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """
    根據鍵點位置和方向，在局部窗口內累積梯度直方圖生成 128 維描述子
    """
    print('Generate Descriptors...')
    descriptors=[]
    for kp in keypoints:
        octv, lyr, scl = unpackOctave(kp)
        img = gaussian_images[octv+1][lyr]
        rows, cols = img.shape
        pt = round(scl*array(kp.pt)).astype(int)
        bins_per_deg = num_bins/360.
        angle = 360.-kp.angle
        cos_a, sin_a = cos(deg2rad(angle)), sin(deg2rad(angle))
        weight_mul = -0.5/((0.5*window_width)**2)
        # 初始化直方圖張量，邊界多兩格
        tensor = zeros((window_width+2, window_width+2, num_bins))
        hist_width = scale_multiplier*0.5*scl*kp.size
        half_w = int(round(hist_width*sqrt(2)*(window_width+1)*0.5))
        half_w = min(half_w, int(sqrt(rows**2+cols**2)))
        # 遍歷窗口內像素
        for dy in range(-half_w, half_w+1):
            for dx in range(-half_w, half_w+1):
                rr = pt[1]+dy; cc = pt[0]+dx
                if rr<=0 or rr>=rows-1 or cc<=0 or cc>=cols-1: continue
                # 旋轉坐標
                r_rot = dx*sin_a + dy*cos_a
                c_rot = dx*cos_a - dy*sin_a
                r_bin = r_rot/hist_width + 0.5*window_width -0.5
                c_bin = c_rot/hist_width + 0.5*window_width -0.5
                if not (-1<r_bin<window_width and -1<c_bin<window_width): continue
                gx = img[rr, cc+1]-img[rr, cc-1]
                gy = img[rr-1, cc]-img[rr+1, cc]
                mag = sqrt(gx*gx+gy*gy)
                orient = rad2deg(arctan2(gy, gx))%360
                w = exp(weight_mul*((r_rot/hist_width)**2+(c_rot/hist_width)**2))
                # 三線性分配到 tensor
                components = [(r_bin,c_bin,w*mag,(orient-angle)*bins_per_deg)]
                for rb,cb,mg,ob in components:
                    r0,c0,o0 = floor([rb,cb,ob]).astype(int)
                    rf,cf,of = rb-r0,cb-c0,ob-o0
                    if o0<0: o0+=num_bins
                    if o0>=num_bins: o0-=num_bins
                    # 四邊分配
                    c1, c0w = mg*rf, mg*(1-rf)
                    c11 = c1*cf; c10 = c1*(1-cf)
                    c01 = c0w*cf; c00 = c0w*(1-cf)
                    # 方向分配
                    tensor[r0+1, c0+1, o0] += c00
                    tensor[r0+1, c0+1,(o0+1)%num_bins] += c01
                    tensor[r0+1, c0+2, o0] += c10
                    tensor[r0+1, c0+2,(o0+1)%num_bins] += c11
        # 去掉邊界，展平
        vec = tensor[1:-1,1:-1,:].flatten()
        # 對比度截斷並正規化
        thr = norm(vec)*descriptor_max_value
        vec[vec>thr] = thr
        vec /= max(norm(vec), float_tolerance)
        # 轉為 0-255 範圍
        vec = round(512*vec); vec[vec<0]=0; vec[vec>255]=255
        descriptors.append(vec)
    return array(descriptors, dtype='float32')
