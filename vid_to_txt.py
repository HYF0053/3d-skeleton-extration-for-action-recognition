import os
import cv2
import tensorflow as tf
import numpy as np
def resize_with_padding(image):
    original_height, original_width = image.shape[:2]
    target_width, target_height = 1080,1920
    # 計算比例並調整大小
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # 創建目標大小的圖像並填充背景
    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left
    color = [0, 0, 0]  # 可以改變填充顏色
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_image
def run_pose_estimation_camera(model_path, video_path, output_path):
    model = tf.saved_model.load(model_path)
    cap = cv2.VideoCapture(video_path)
    poses3d_all_frames = []
    count = 0    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #圖片改變大小依照情況使用，避免形變造成辨識結果出問題
        #frame = resize_with_padding(frame)    
        frame = cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_AREA)
        # Convert frame to tensor
        image = tf.convert_to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Perform pose detection
        pred = model.detect_poses(image, skeleton='smpl_24')
        poses3d = pred['poses3d'].numpy()
        if(poses3d.shape[0]!=2):
            continue     
        poses3d_all_frames.append(poses3d)
        #print(f"Frame {count}: {poses3d}")
        print('frame:' + str(count))
        count += 1    
    cap.release()
    cv2.destroyAllWindows()
    # Save the results as a .npy file
    np.save(output_path, poses3d_all_frames)
def process_all_videos_in_folder(model_path, folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)    
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):
            video_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_poses3d.npy")
            print(f'Processing {video_path}...')
            run_pose_estimation_camera(model_path, video_path, output_path)
            print(f'Finished processing {video_path}, results saved to {output_path}')
if __name__ == '__main__':
    # 設定模型路徑和資料夾路徑，一次只處理一個資料夾，要處理多個類別資料夾程式須在調整
    model_path = 'metrabs_mob3l_y4t'
    folder_path = input('folder_path:')
    output_folder = input('output_folder:')  
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 批次處理資料夾中的所有影片
    process_all_videos_in_folder(model_path, folder_path, output_folder)
