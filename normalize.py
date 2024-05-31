import numpy as np
import os
import sys
from shutil import copyfile
import matplotlib.pyplot as plt
from scipy.stats import norm
#import argparse


def normalize_along_axis(data, axis):
    # 计算沿指定轴的最小值和最大值
    min_val = np.min(data[:,:,axis,:])
    max_val = np.max(data[:,:,axis,:])
    #print(data[:,:,axis,:])
    print('>>>>>'+str(min_val)+','+str(max_val))
    # 计算差值
    range_val = max_val - min_val
    
    # 检查是否存在最大值和最小值相等的情况
    #range_val[range_val == 0] = 1  # 避免除以零
    
    # 执行归一化
    normalized_data = (data[:,:,axis,:]- min_val) / range_val
    return normalized_data 
    
def read_all_npy_files_in_folder(folder_path,output_folder):
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            print(f'Reading {file_path}...')
            data = np.load(file_path, allow_pickle=True)  
            for axis in range(data.shape[2]):
                #print(str(axis))
                data[:,:,axis,:] = normalize_along_axis(data, axis)
            output_path = output_folder+'/'+str(count)+'.npy'   
            np.save(output_path, data) 
            print(data)
            count+=1   
                
if __name__ == '__main__':
    np_folder = input('folder_path:')
    output_folder = input('output_path:')  
    if not os.path.exists(output_folder):
        os.makedirs(output_folder) 
    read_all_npy_files_in_folder(np_folder,output_folder)

    
