import os
import numpy as np
from sklearn.model_selection import train_test_split

def pad_or_truncate(data, fixed_length):
    """
    Pad or truncate the input data to the fixed length.
    """
    if data.shape[0] > fixed_length:
        return data[:fixed_length]
    elif data.shape[0] < fixed_length:
        pad_width = [(0, fixed_length - data.shape[0])] + [(0, 0)] * (len(data.shape) - 1)
        return np.pad(data, pad_width, mode='constant')
    else:
        return data

def load_data_from_folder(folder_path, label):
    data = []
    labels = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            npy_data = np.load(file_path)
            npy_data = pad_or_truncate(npy_data, 800)
            npy_data = npy_data.transpose(3, 0, 2, 1)
            print(file_path + str(npy_data.shape))
            data.append(npy_data)
            labels.append(label)
    return data, labels

def load_data_from_all_folders(parent_folder):
    data = []
    labels = []
    label = 0
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            folder_data, folder_labels = load_data_from_folder(folder_path, label)
            data.extend(folder_data)
            labels.extend(folder_labels)
            label += 1
    return data, labels

def split_and_save_npz(parent_folder, output_path, test_size=0.3):
    # Load data from all class folders
    all_data, all_labels = load_data_from_all_folders(parent_folder)

    # Convert to numpy arrays
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)

    # Split into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=test_size, random_state=42)
    print(all_data.shape)

    # Save to .npz file
    np.savez(output_path, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
    print(f"Data saved to {output_path}")

if __name__ == '__main__':
    parent_folder = input('folder_path:')
    output_path = input('output_path:')  
    if not os.path.exists(output_path):
        os.makedirs(output_path) 
    # Create the combined npz file
    split_and_save_npz(parent_folder, output_path)
