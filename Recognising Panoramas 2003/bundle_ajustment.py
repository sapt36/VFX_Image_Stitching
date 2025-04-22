# bundle_adjustment.py
import cv2
import numpy as np
from scipy.optimize import least_squares
from image_matching import match_and_verify_images
from feature_matching import extract_sift_features

def collect_feature_matches(image_paths):
    keypoints_list = []
    descriptors_list = []
    for path in image_paths:
        img = cv2.imread(path)
        kp, desc = extract_sift_features(img)
        keypoints_list.append(kp)
        descriptors_list.append(desc)
    # 先做匹配+驗證，取得匹配影像對
    matched_pairs = match_and_verify_images(descriptors_list, keypoints_list, m=6)
    # 再依 (i,j) 重新取得 inlier feature 對
    flann = cv2.FlannBasedMatcher(dict(algorithm=1,trees=5), dict(checks=50))
    matches_for_ba = []
    for i,j in matched_pairs:
        # 重算 KNN + ratio，再 RANSAC 取得 mask
        knn = flann.knnMatch(descriptors_list[i], descriptors_list[j], k=2)
        good = [m for m,n in knn if m.distance < 0.75*n.distance]
        if len(good)<4: continue
        pts_i = np.float32([keypoints_list[i][m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pts_j = np.float32([keypoints_list[j][m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, 3.0)
        inliers = mask.ravel().nonzero()[0]
        pairs = []
        for idx in inliers:
            ui = tuple(pts_i[idx,0])
            uj = tuple(pts_j[idx,0])
            pairs.append((ui, uj))
        matches_for_ba.append((i, j, pairs))
    return keypoints_list, descriptors_list, matches_for_ba

# 將旋轉向量與焦距 pack/unpack
def pack_params(rvecs, focals):
    return np.hstack([r.flatten() for r in rvecs] + [focals])

def unpack_params(x, n):
    rvecs = [x[3*i:3*i+3].reshape(3,1) for i in range(n)]
    focals = x[3*n:]
    return rvecs, focals

def project(pt, rvec, f, K0):
    R,_ = cv2.Rodrigues(rvec)
    K = K0.copy()
    K[0,0] = K[1,1] = f
    uvw = K.dot(R.dot(np.array([pt[0],pt[1],1.0]).reshape(3,1)))
    u,v,w = uvw.flatten()
    return np.array([u/w, v/w])

# 殘差函式
def residuals(x, n, matches, K0):
    rvecs, focals = unpack_params(x, n)
    res = []
    for i,j,pairs in matches:
        for ui, uj in pairs:
            # 先反投影到 z=1 假深度
            P = np.array([ui[0], ui[1], 1.0])
            # i→j→i 投影殘差
            pj = project(P, rvecs[j], focals[j], K0)
            pji= project(np.array([pj[0],pj[1],1.0]), rvecs[i], focals[i], K0)
            diff = np.array(ui) - pji
            # 魯棒截斷 xmax=1 px
            norm2 = np.linalg.norm(diff)
            w = min(norm2, 1.0)
            res.extend((w*diff).tolist())
    return np.array(res)

if __name__ == "__main__":
    # 影像清單
    image_paths = [f"../test_images/prtn{idx:02d}.jpg" for idx in range(18)]
    keypoints_list, descriptors_list, matches_for_ba = collect_feature_matches(image_paths)

    # 相機參數初始值：所有 R=I (rvec=0)，focals=影像寬度
    n = len(image_paths)
    rvecs0 = [np.zeros((3,1),dtype=float) for _ in range(n)]
    img0 = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    f0 = img0.shape[1]  # 初始焦距
    focals0 = np.array([f0]*n, dtype=float)

    # 建立 K0（假設主點在影像中心）
    cx, cy = img0.shape[1]/2, img0.shape[0]/2
    K0 = np.array([[f0,0,cx],[0,f0,cy],[0,0,1]], dtype=float)

    x0 = pack_params(rvecs0, focals0)
    # 5. 執行 Levenberg–Marquardt
    result = least_squares(
        residuals, x0,
        args=(n, matches_for_ba, K0),
        method='lm',
        ftol=1e-6, xtol=1e-6, gtol=1e-6,
        loss='huber', verbose=2
    )

    # 6. 解包並顯示結果
    rvecs_opt, focals_opt = unpack_params(result.x, n)
    for i,(rv,f) in enumerate(zip(rvecs_opt, focals_opt)):
        print(f"Image {i}: rvec = {rv.ravel()}, focal = {f:.2f}")
