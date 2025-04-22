import cv2

# 1. 匯入 SIFT 實作
from sift_impl import compute_keypoints_and_descriptors

# 2. 匯入影像匹配與驗證的函式
from image_matching import match_and_verify_images, find_panoramas

def extract_sift_features(img):
    """
    使用純 Python SIFT 實作計算 keypoints 和 descriptors
    """
    # 如為彩色影像，先轉灰階
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 呼叫自訂的 SIFT 演算法
    keypoints, descriptors = compute_keypoints_and_descriptors(img)
    return keypoints, descriptors

def match_features(desc1, desc2, k=4):
    """
    兩影像之間的 KNN + Lowe ratio 測試匹配（僅示範用，可選擇省略）
    """
    index_params  = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    knn_matches = flann.knnMatch(desc1, desc2, k=k)
    good = []
    for nbrs in knn_matches:
        if len(nbrs) < 2:
            continue
        m, n = nbrs[0], nbrs[1]
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

if __name__ == "__main__":
    # 一組要拼接的影像路徑列表
    image_paths = [
        "../test_images/prtn00.jpg",
        "../test_images/prtn01.jpg",
        "../test_images/prtn02.jpg",
        "../test_images/prtn03.jpg",
        "../test_images/prtn04.jpg",
        "../test_images/prtn05.jpg",
        "../test_images/prtn06.jpg",
        "../test_images/prtn07.jpg",
        "../test_images/prtn08.jpg",
        "../test_images/prtn09.jpg",
        "../test_images/prtn10.jpg",
        "../test_images/prtn11.jpg",
        "../test_images/prtn12.jpg",
        "../test_images/prtn13.jpg",
        "../test_images/prtn14.jpg",
        "../test_images/prtn15.jpg",
        "../test_images/prtn16.jpg",
        "../test_images/prtn17.jpg"
    ]

    keypoints_list   = []
    descriptors_list = []

    # --- 1. 提取所有影像的 SIFT 特徵 ---
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"無法讀取影像：{path}")
            continue
        kp, desc = extract_sift_features(img)
        print(f"{path} 共有 {len(kp)} 個 keypoints")
        keypoints_list.append(kp)
        descriptors_list.append(desc)

    # --- 2. （示範）單對單匹配可視化 ---
    if len(keypoints_list) >= 2:
        matches01 = match_features(descriptors_list[0], descriptors_list[1], k=4)
        print(f"影像 0 ↔ 1 有 {len(matches01)} 個良好匹配")
        vis = cv2.drawMatches(
            cv2.imread(image_paths[0]), keypoints_list[0],
            cv2.imread(image_paths[1]), keypoints_list[1],
            matches01, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imshow("Pairwise Matches", vis)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    # --- 3. 批次匹配 + RANSAC + 概率驗證 ---
    #    讓 descriptors_list, keypoints_list 傳入 match_and_verify_images
    matches   = match_and_verify_images(descriptors_list, keypoints_list, m=6)
    panoramas = find_panoramas(matches)

    # 列印通過驗證的影像對
    print("\n=== 通過驗證的影像對 (i, j) ===")
    for i, j in matches:
        print(f"  影像 {i} ↔ 影像 {j}")

    # 列印偵測到的全景序列
    print("\n=== 偵測到的全景序列 ===")
    for idx, pano in enumerate(panoramas):
        print(f"  全景 {idx}: 影像序號 {pano}")
