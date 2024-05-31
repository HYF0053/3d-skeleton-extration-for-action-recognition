# 3d-skeleton-extration-for-action-recognition
本程式只做簡單提取，輸入一段影片，輸出所看到的三維骨架序列，並將其轉換成可以訓練的npz檔，train和test已分割好  
骨架主要參考自：[MeTRAbs: Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose Estimation](https://github.com/isarandi/metrabs?tab=readme-ov-file#metrabs-absolute-3d-human-pose-estimator)  
# 環境安裝
先clone整個MeTRAbs檔案，並把MeTRAbs所需的環境安裝好，能夠運行原論文的demo，模型權重請參考[連結](https://omnomnom.vision.rwth-aachen.de/data/metrabs/)  
安裝opencv-python，並將所有本專案中的.py檔放入MeTRAbs資料夾中即可運行
# 程式功能及運行
