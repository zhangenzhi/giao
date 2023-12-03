import random
import fnmatch, os
import numpy as np
import tensorflow as tf
from tensorflow.io import gfile
from scipy import io as scipy_io
import matplotlib.pyplot as plt

import os
import json
import tempfile
import string
import ruamel.yaml as yaml

def display(display_list, save_name=None, show=False):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'Predicted Image', 'Test Image']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    if save_name!=None:
        plt.savefig("./result/{}.png".format(save_name))
        
    if show:
        plt.show()
        
    plt.close()

def plot_train(display_list, save_name=None, show=False):
    # plt.figure(figsize=(10, 10))
    label = ['Train', 'Test']
    for i in range(len(display_list)):
        plt.plot(display_list[i], label=label[i])
        plt.legend()
        
    if save_name!=None:
        plt.savefig("./result/{}.png".format(save_name))
        
    if show:
        plt.show()
  
def check_mkdir(path):
    if not os.path.exists(path=path):
        print("no such path: {}, but we made.".format(path))
        os.makedirs(path)

def glob_tfrecords(input_dirs, glob_pattern="example", recursively=False):
    file_path_list = []
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]
    for root_path in input_dirs:
        assert gfile.exists(root_path), "{} does not exist.".format(root_path)
        if not gfile.isdir(root_path):
            file_path_list.append(root_path)
            continue
        if not recursively:
            for filename in gfile.listdir(root_path):
                if fnmatch.fnmatch(filename, glob_pattern):
                    file_path_list.append(os.path.join(root_path, filename))
        else:
            for dir_path, _, filename_list in gfile.walk(root_path):
                for filename in filename_list:
                    if fnmatch.fnmatch(filename, glob_pattern):
                        file_path_list.append(os.path.join(dir_path, filename))
    return file_path_list

def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images = (train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    
    return train_images, test_images

def check_mkdir(path):
    if not os.path.exists(path=path):
        print("no such path: {}, but we made.".format(path))
        os.makedirs(path)


def get_yml_content(file_path):
    '''Load yaml file content'''
    try:
        with open(file_path, 'r') as file:
            return yaml.load(file, Loader=yaml.Loader)
    except yaml.scanner.ScannerError as err:
        print('yaml file format error!')
        print(err)
        exit(1)
    except Exception as exception:
        print(exception)
        exit(1)
        
def save_yaml_contents(file_path, contents):
    try:
        with open(file_path, 'w') as file:
            yaml.dump(contents, file)
    except Exception as exception:
        print(exception)
        exit(1)
        


def get_json_content(file_path):
    '''Load json file content'''
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except TypeError as err:
        print('json file format error!')
        print(err)
        return None

def generate_temp_dir():
    '''generate a temp folder'''

    def generate_folder_name():
        return os.path.join(
            tempfile.gettempdir(), 'tki',
            ''.join(random.sample(string.ascii_letters + string.digits, 8)))

    temp_dir = generate_folder_name()
    while os.path.exists(temp_dir):
        temp_dir = generate_folder_name()
    os.makedirs(temp_dir)
    return temp_dir


def write_log(file, msg):
    with open(file, 'a+') as f:
        f.write(msg)
        f.write('\n')
