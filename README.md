# 3d-skeleton-extration-for-action-recognition
本程式只做簡單提取，輸入一段影片，輸出所看到的三維骨架序列，並將其轉換成可以訓練的npz檔，train和test已分割好  
骨架主要參考自：[MeTRAbs: Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose Estimation](https://github.com/isarandi/metrabs?tab=readme-ov-file#metrabs-absolute-3d-human-pose-estimator)  
# 環境安裝
先clone整個MeTRAbs檔案，並把MeTRAbs所需的環境安裝好，能夠運行原論文的demo，模型權重請參考[連結](https://omnomnom.vision.rwth-aachen.de/data/metrabs/)  
安裝opencv-python，並將所有本專案中的.py檔放入MeTRAbs資料夾中即可運行
# 程式功能及運行
test_screen.py執行後，會在偵測螢幕上的部份畫面，辨識畫面中的骨架，只須將圖片或影片拉到特定區域，即可快速查看骨架提取效果    
test_vid.py執行後，會讀取影片並將骨架圖儲存下來查看結果  
vid_to_txt.py執行後會將每一個影片的骨架序列儲存成單獨的npy檔，方便視覺化跟後續dataset處理  
normalize.py會將每一個npy檔在xyz軸座標分別做歸一化至0～1之間  
transform.py會將所有類別分割為train和test
