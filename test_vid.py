import cv2
import time
import os
import mss
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

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
def run_pose_estimation_video(model_path,vid_path,out_name):
    model = tf.saved_model.load(model_path)
    cap = cv2.VideoCapture(vid_path)
    count = 0
    out_path = './result/'+out_name
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    #poses3d_all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_with_padding(frame)
        # Convert frame to tensor
        image = tf.convert_to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Perform pose detection
        pred = model.detect_poses(image, skeleton='smpl_24')
        poses3d = pred['poses3d'].numpy()               
        if(poses3d.shape[0]!=2):
            continue                      
        #poses3d_all_frames.append(poses3d)    
        print(count)
        print(poses3d)
        # Visualize the result       
        visualize(
            frame, 
            pred['boxes'].numpy(),
            pred['poses3d'].numpy(),
            pred['poses2d'].numpy(),
            model.per_skeleton_joint_edges['smpl_24'].numpy(),
            count,
            out_path )       
        count +=1        
    cap.release()
    cv2.destroyAllWindows()

def visualize(image, detections, poses3d, poses2d, edges,count,out_path):
    fig = plt.figure(figsize=(10, 5.2))
    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.imshow(image)
    for x, y, w, h in detections[:, :4]:
        image_ax.add_patch(Rectangle((x, y), w, h, fill=False))
    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
    pose_ax.view_init(5, -85)
    pose_ax.set_xlim3d(-1500, 1500)
    pose_ax.set_zlim3d(-1500, 1500)
    pose_ax.set_ylim3d(0, 3000)
    # Matplotlib plots the Z axis as vertical, but our poses have Y as the vertical axis.
    # Therefore, we do a 90° rotation around the X axis:
    poses3d[..., 1], poses3d[..., 2] = poses3d[..., 2], -poses3d[..., 1]
    for pose3d, pose2d in zip(poses3d, poses2d):
        for i_start, i_end in edges:
            image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
            pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=2)
        image_ax.scatter(*pose2d.T, s=2)
        pose_ax.scatter(*pose3d.T, s=2)

    fig.tight_layout()
    #plt.show()
    plt.savefig(out_path +'/'+str(count)+'.png')


# 调用函数
if __name__ == '__main__':
    vid = input('video name:')
    out = input('save file name :')    
    run_pose_estimation_video('metrabs_mob3l_y4t',vid,out)  # 传入模型路径
