# 3d-skeleton-extration-for-action-recognition
本程式只做簡單提取，輸入一段影片，輸出所看到的三維骨架序列，並將其轉換成可以訓練的npz檔，train和test已分割好  
骨架主要參考自：[MeTRAbs: Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose Estimation](https://github.com/isarandi/metrabs?tab=readme-ov-file#metrabs-absolute-3d-human-pose-estimator)  
# 環境安裝
先clone整個MeTRAbs檔案，並把MeTRAbs所需的環境安裝好，能夠運行原論文的demo，模型權重請參考[連結](https://omnomnom.vision.rwth-aachen.de/data/metrabs/)  
將所有本專案中的.py檔放入MeTRAbs資料夾中即可運行
# 程式功能及運行
## test_screen.py
執行後，會在偵測螢幕上的部份畫面，辨識畫面中的骨架，只須將圖片或影片拉到特定區域，即可快速查看骨架提取效果    
  
## test_vid.py
執行後，會讀取影片並將骨架圖儲存下來，查看結果是否符合需求  
  
## vid_to_txt.py
執行後會讀取每一個資料夾中個.mp4影片（如須其他格式請直接更改程式），處理後的骨架序列會儲存成單獨的npy檔，大小為（c,t,j,m），一個npy代表一個影片，方便視覺化跟後續dataset處理 ，程式可以設定超過或不等於幾個人就不儲存，固定輸出的大小一致，預設為兩人，程式38行中修改 
  
## normalize.py
讀取一個資料中所有的npy檔，將每一個npy檔在xyz軸座標分別做歸一化至0～1之間 
  
## transform.py
將讀取一個資料夾中的所有子資料夾類別資料，將所有類別合併分割為train和test，前處理會將其時間長度padding或減少至相同，存成單一npz檔(b,c,t,j,m)  

## 除了transform.py可以一次處理多個類別資料夾外，其他都只能處理單一類別資料夾，如要整批處理需要在修改程式  
