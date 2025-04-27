# Digital-Visual-Effects---Image-Stitching

## code:
1. `image_stitching_harris.py`
2. `image_stitching_sift.py`
3. `sift_impl.py`
4. `sift_visualizeUI.py`
```bash
pip install opencv-python numpy matplotlib PyQt5
```
---

### 1. `image_stitching_harris.py`

**Dependencies**：
- `opencv-python`
- `numpy`
- `math`
- `time`
- （標準Python內建 `os`）

**安裝指令**：
```bash
pip install opencv-python numpy
```

**Run Code**：
```bash
python image_stitching_harris.py
```
程式啟動後會要求輸入**圖片資料夾路徑**，資料夾內應有一個 `pano.txt`，裡面寫每張圖片路徑和焦距。 (由 autostitch 生成即可)

---

### 2. `image_stitching_sift.py`

**Dependencies**：
- `opencv-python`
- `numpy`
- `math`
- `time`
- 自製模組：`sift_impl.py`
- （標準Python內建 `os`）

**安裝指令**：
```bash
pip install opencv-python numpy
```

**Run Code**：
```bash
python image_stitching_sift.py
```
同樣，啟動後會問你**圖片資料夾位置**，要有 `pano.txt`。

---

### 3. `sift_impl.py`

（功能模組，無需執行）

**Dependencies**：
- `opencv-python`
- `numpy`
- （標準Python內建 `functools`）

**安裝指令**：
```bash
pip install opencv-python numpy
```


「  `image_stitching_sift.py` 和 `sift_visualizeUI.py` 會使用到 」

---

### 4. `sift_visualizeUI.py`

**Dependencies**：
- `opencv-python`
- `numpy`
- `matplotlib`
- `PyQt5`
- 自製模組：`sift_impl.py`

**安裝指令**：
```bash
pip install opencv-python numpy matplotlib PyQt5
```

**Run Code**：
```bash
python sift_visualizeUI.py
```
執行後會打開一個 **PyQt 視窗介面**，讓你可視化：
- 基底影像
- 高斯金字塔
- DoG 金字塔
- 特徵點分布
- 描述子向量
- 特徵匹配結果
```
預設使用 parrington/prtn00.jpg, prtn01.jpg 做示例
```
---

### 總整理

| 程式                     | 主要功能         | 需安裝套件                                | 是否直接執行 |
|:-------------------------|:-------------|:------------------------------------------|:------------|
| image_stitching_harris.py | Harris 拼接影像  | opencv-python, numpy                      | ✅ |
| image_stitching_sift.py   | SIFT 拼接影像    | opencv-python, numpy                      | ✅ |
| sift_impl.py              | SIFT 演算法模組   | opencv-python, numpy                      | ⛔|
| sift_visualizeUI.py       | SIFT 流程可視化介面 | opencv-python, numpy, matplotlib, PyQt5   | ✅ |

---

  
