# Digital-Visual-Effects---Image-Stitching

## code:
1. `image_stitching_harris.py`
2. `image_stitching_sift.py`
3. `sift_impl.py`
4. `sift_visualizeUI.py`
5. `harris_visualizeUI.py`

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
程式啟動後會要求輸入**圖片資料夾路徑**及結束前要求輸入**裁切邊界大小**，圖片資料夾內應有一個 `pano.txt`(由 autostitch 生成)，裡面寫每張圖片路徑和焦距。
執行範例如下：
```bash
C:\Users\853uj\anaconda3\python.exe C:\Users\853uj\PyCharmProject\DVE_HW2\image_stitching_harris.py 
請輸入圖片資料夾位置 (預設為 .) ：C:\Users\853uj\PyCharmProject\DVE_HW2\out
請輸入 pano.txt 檔案路徑 (在圖片資料夾內可直接按enter)：
已從 pano.txt 讀取 2 張影像路徑及其焦距。
圓柱投影完成，總共 2 張影像。
Timer: 0.59 秒 讀取影像、圓柱投影
拼接中：第 1 / 1 張...
Timer: 0.74 秒 Harris角點 + RANSAC 完成
實際拼接：第 1 / 1 張...
請輸入裁切邊界 (預設 15)：30
全景拼接完成，輸出：C:\Users\853uj\PyCharmProject\DVE_HW2\out/panoroma_harris.jpg
總共花費 4.32 秒

Process finished with exit code 0
```
裁切邊界大小建議
- `out` = 30
- `parrington` = 15
- `grail` = 17

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
同樣，啟動後會問你**圖片資料夾位置**，要有 `pano.txt`，結束前會問你**裁切邊界大小**，可參考以下範例。
```bash
C:\Users\853uj\anaconda3\python.exe C:\Users\853uj\PyCharmProject\DVE_HW2\image_stitching_sift.py 
請輸入圖片資料夾位置 (預設為 .) ：C:\Users\853uj\PyCharmProject\DVE_HW2\out
請輸入 pano.txt 檔案路徑 (若同資料夾僅輸入檔名)：
已從 pano.txt 讀取 2 張影像路徑及其焦距。
圓柱投影完成，總共 2 張影像。
Timer: 0.60 秒 讀取影像、圓柱投影、計算drift
拼接中：第 1 / 1 張...
Timer: 50.70 秒 SIFT運算
實際拼接：第 1 / 1 張...
請輸入裁切邊界 (預設 15)：30
全景拼接完成，輸出：C:\Users\853uj\PyCharmProject\DVE_HW2\out/panoroma_sift.jpg
總共花費 85.18 秒

Process finished with exit code 0
```

裁切邊界大小建議
- `out` = 30
- `parrington` = 15
- `grail` = 17

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

### 5. `harris_visualizeUI.py`

**Dependencies**：
- `opencv-python`
- `numpy`
- `PyQt5`

**安裝指令**：
```bash
pip install opencv-python numpy PyQt5
```

**Run Code**：
```bash
python harris_visualizeUI.py
```
執行後會打開一個 **PyQt 視窗介面**，讓你可視化：Harris 特徵匹配

```
可選擇輸入你想匹配的圖片
```
---

### 總整理

| 程式                        | 主要功能            | 需安裝套件                                   | 是否直接執行 |
|:--------------------------|:----------------|:----------------------------------------|:------------|
| image_stitching_harris.py | Harris 拼接影像     | opencv-python, numpy                    | ✅ |
| image_stitching_sift.py   | SIFT 拼接影像       | opencv-python, numpy                    | ✅ |
| sift_impl.py              | SIFT 演算法模組      | opencv-python, numpy                    | ⛔|
| sift_visualizeUI.py       | SIFT 流程可視化介面    | opencv-python, numpy, matplotlib, PyQt5 | ✅ |
| harris_visualizeUI.py     | Harris特徵匹配可視化介面 | opencv-python, numpy, PyQt5             | ✅ |

---

  
