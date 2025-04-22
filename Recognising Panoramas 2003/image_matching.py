import cv2
import numpy as np
from collections import defaultdict, deque


def match_and_verify_images(descriptors_list, keypoints_list,
                            m=6, p1=0.7, p0=0.01, pmin=0.97):
    """
    對 N 張影像的 descriptors 和 keypoints 執行：
      1. FLANN 建立所有影像對的 knnMatch，並做 Lowe ratio test
      2. 取每張影像匹配數最多的前 m 名作候選
      3. 對每對候選影像，用 RANSAC 估算 homography，獲得內點數 ni 和總匹配數 nf
      4. 若 ni > 5.9 + 0.22*nf，則接受此匹配
    回傳：通過驗證的 (i, j) 匹配對清單
    """
    num_images = len(descriptors_list)
    # 1. 計算所有影像對的初步匹配
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),  # KD-Tree
        dict(checks=50)
    )
    all_matches = {}  # key=(i,j), value=[DMatch,...]
    for i in range(num_images):
        desc_i = descriptors_list[i]
        if desc_i is None: continue
        for j in range(i+1, num_images):
            desc_j = descriptors_list[j]
            if desc_j is None: continue
            knn = flann.knnMatch(desc_i, desc_j, k=2)
            # Lowe ratio test
            good = [m for m, n in knn if m.distance < 0.75*n.distance]
            all_matches[(i, j)] = good

    # 2. 選前 m 名候選
    candidates = defaultdict(list)
    for (i, j), matches in all_matches.items():
        candidates[i].append((j, len(matches)))
        candidates[j].append((i, len(matches)))
    for k, lst in candidates.items():
        lst.sort(key=lambda x: -x[1])
        candidates[k] = [ij for ij,_ in lst[:m]]

    # 3. RANSAC + 概率驗證
    accepted = []
    for i in range(num_images):
        for j in candidates[i]:
            # 只處理 i<j 避免重複
            if i >= j or (i,j) not in all_matches:
                continue
            matches = all_matches[(i,j)]
            nf = len(matches)
            if nf < 4:
                continue
            # 構造點集
            pts_i = np.float32([ keypoints_list[i][m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            pts_j = np.float32([ keypoints_list[j][m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
            H, mask = cv2.findHomography(pts_i, pts_j, cv2.RANSAC, 3.0)
            if mask is None:
                continue
            ni = int(mask.sum())
            # 4. 驗證條件 ni > 5.9 + 0.22*nf
            if ni > 5.9 + 0.22 * nf:
                accepted.append((i,j))

    return accepted


def find_panoramas(matched_pairs):
    """
    根據 matched_pairs（(i,j) 列表），
    找出所有連通子圖，每個子圖即為一個全景序列。
    """
    graph = defaultdict(list)
    for i,j in matched_pairs:
        graph[i].append(j)
        graph[j].append(i)

    visited = set()
    panoramas = []
    for node in graph:
        if node not in visited:
            queue = deque([node])
            visited.add(node)
            comp = []
            while queue:
                u = queue.popleft()
                comp.append(u)
                for v in graph[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
            panoramas.append(sorted(comp))
    return panoramas


