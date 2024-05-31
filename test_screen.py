import cv2
import time
import mss
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

def run_pose_estimation_camera(model_path):
    model = tf.saved_model.load(model_path)
    count = 0
    paused = False
    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {"top": 40, "left": 0, "width": 800, "height": 640}  #螢幕所看的位置      
        plt.ion()  # 開啟交互模式
        fig, (image_ax, pose_ax) = plt.subplots(1, 2, figsize=(10, 5.2))
        pose_ax = fig.add_subplot(1, 2, 2, projection='3d')        
        while True:
            if not paused:
                # Get raw pixels from the screen, save it to a Numpy array
                img = np.array(sct.grab(monitor))
                cv2.imshow('frame', img)
                # Display the picture
                frame = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
                image = tf.convert_to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pred = model.detect_poses(image, skeleton='smpl_24')               
                #print(count) #幀數
                #print(pred['poses3d'].numpy) #3d座標
                count += 1                
                # Visualize the result
                visualize(
                    frame, 
                    pred['boxes'].numpy(),
                    pred['poses3d'].numpy(),
                    pred['poses2d'].numpy(),
                    model.per_skeleton_joint_edges['smpl_24'].numpy(),
                    image_ax, pose_ax)                
                fig.tight_layout()
                plt.draw()                
                plt.pause(0.001)  # 暫停以便更新畫面            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused  # 按 'p' 鍵切換暫停和繼續
        plt.ioff()  # 關閉交互模式
        plt.show()  # 顯示所有累積的圖像
    cv2.destroyAllWindows()

def visualize(image, detections, poses3d, poses2d, edges, image_ax, pose_ax):
    image_ax.clear()  # 清除當前圖像
    pose_ax.clear()   # 清除當前3D圖像
    image_ax.imshow(image)
    for x, y, w, h in detections[:, :4]:
        image_ax.add_patch(Rectangle((x, y), w, h, fill=False))
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

if __name__ == '__main__':
    run_pose_estimation_camera('metrabs_mob3l_y4t')  # 传入模型路径
