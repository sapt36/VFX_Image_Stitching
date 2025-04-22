import cv2

# 匯入 SIFT 實作
from sift_impl import compute_keypoints_and_descriptors


def extract_sift_features(img):
    """
    使用純 Python 實作計算 keypoints 與 descriptors，取代 OpenCV SIFT
    """
    # 確保為灰階 float32 圖片
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = compute_keypoints_and_descriptors(img)
    return keypoints, descriptors


def match_features(desc1, desc2, k=4):
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    knn_matches = flann.knnMatch(desc1, desc2, k=k)

    good_matches = []
    for neighbours in knn_matches:
        # 只要最少有两项，就取最好的两个做 ratio test
        if len(neighbours) < 2:
            continue
        m, n = neighbours[0], neighbours[1]
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches



if __name__ == "__main__":
    # 讀取兩張影像
    img1 = cv2.imread("../test_images/prtn01.jpg")
    img2 = cv2.imread("../test_images/prtn00.jpg")
    if img1 is None or img2 is None:
        print("無法讀取輸入影像，請檢查路徑。")
        exit(1)

    # 提取 SIFT 特徵
    kp1, desc1 = extract_sift_features(img1)
    kp2, desc2 = extract_sift_features(img2)
    print(f"Image1: {len(kp1)} keypoints, Image2: {len(kp2)} keypoints")

    # 匹配 descriptors
    matches = match_features(desc1, desc2, k=4)
    print(f"找到 {len(matches)} 個良好匹配")

    # 畫出匹配結果
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("SIFT Matches", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
