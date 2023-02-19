import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict
from tensorflow.keras.layers.experimental import preprocessing

from utiliz import normalization
    
class Cifar10DataLoader:
    def __init__(self, dataloader_args):
        self.dataloader_args = dataloader_args
        self.info = edict({'train_size':50000,'test_size':10000,'image_size':[32,32,3],
                            'train_step': int(50000/dataloader_args['batch_size']),
                            'valid_step': int(10000/dataloader_args['batch_size']),
                            'test_step': int(10000/dataloader_args['batch_size']),
                            'epochs': dataloader_args['epochs']})
        
    def load_dataset(self, epochs=-1, format=None):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        x_train = (x_train / 255.0).astype(np.float32)
        y_train = y_train.astype(np.float32)
        
        x_test = (x_test / 255.0).astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        # x_train = np.reshape(x_train, [50000,32,32,3])
        # x_test = np.reshape(x_test, [10000,32,32,3])
        # if self.dataloader_args['da']:
        x_train,x_test = normalization(x_train, x_test)
        
        #on-hot
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        data_augmentation = tf.keras.Sequential([
                    preprocessing.RandomContrast(0.1),
                    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
                    preprocessing.RandomCrop(32, 32),
                    preprocessing.RandomZoom(0.1)
                    ])

        full_size = len(x_train)
        test_size = len(x_test)
        
        train_size = int(1.0 * full_size)
        valid_size = int(0.5 * test_size)

        full_dataset = tf.data.Dataset.from_tensor_slices({'inputs': x_train, 'labels': y_train})
        full_dataset = full_dataset.shuffle(full_size)

        train_dataset = full_dataset.take(train_size)
        train_dataset = train_dataset.batch(self.dataloader_args['batch_size'], drop_remainder=True)
        # data augmentation
        if self.dataloader_args['da']:
            train_dataset = train_dataset.map(lambda x:{'inputs':data_augmentation(x['inputs']),'labels': x['labels']}, num_parallel_calls=16)
        train_dataset = train_dataset.prefetch(1)
        train_dataset = train_dataset.repeat(epochs)


        # valid_dataset = full_dataset.skip(train_size)
        # valid_dataset = valid_dataset.take(valid_size).repeat(epochs)
        # valid_dataset = valid_dataset.batch(self.dataloader_args['batch_size'])

        test_dataset = tf.data.Dataset.from_tensor_slices({'inputs': x_test, 'labels': y_test})
        test_dataset = test_dataset.shuffle(test_size)
        # valid_dataset = test_dataset.take(valid_size).batch(self.dataloader_args['batch_size']).repeat(epochs)
        # test_dataset = test_dataset.skip(valid_size).batch(self.dataloader_args['batch_size']).repeat(epochs)
        test_dataset = test_dataset.batch(self.dataloader_args['batch_size'], drop_remainder=True).repeat(epochs)
        valid_dataset = test_dataset

        return train_dataset, valid_dataset, test_dataset


class MnistDataLoader:
    
    def __init__(self, dataloader_args):
        self.dataloader_args = dataloader_args
        self.info = edict({'train_size':50000,'test_size':10000,'image_size':[28, 28, 1],
                'train_step': int(50000/dataloader_args['batch_size']),
                'valid_step': int(10000/dataloader_args['batch_size']),
                'test_step': int(10000/dataloader_args['batch_size']),
                'epochs': dataloader_args['epochs']})
        
    def load_datadict(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        x_train = (x_train / 255.0).astype(np.float32)
        x_test = (x_test / 255.0).astype(np.float32)
       
        
        x_train = np.reshape(x_train, [60000,28,28,1])
        x_test = np.reshape(x_test, [10000,28,28,1])
        if self.dataloader_args['da']:
            x_train, x_test = normalization(x_train, x_test)
        
        train_dict = {}
        test_dict = {}
        keys = list(set(y_test))
        for k in keys:
            train_dict[str(k)] = []
            test_dict[str(k)] = []
            
        for idx in range(len(x_train)):
            train_dict[str(y_train[idx])].append(x_train[idx])
            
        for idx in range(len(x_test)):
            test_dict[str(y_test[idx])].append(x_test[idx])
        
        # datasets = []
        # for k in train_dict.keys():
        #     labels = [np.identity(10)[int(float(k))] for i in range(len(train_dict[k]))]
        #     datasets.append(tf.data.Dataset.from_tensor_slices({"inputs":train_dict[k],'labels': labels}))
        # train_set = tf.data.Dataset.sample_from_datasets(datasets, weights=[1/len(keys) for i in range(len(keys))])
        return train_dict, test_dict
    
    def load_dataset(self, epochs=-1, format=None):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        x_train = (x_train / 255.0).astype(np.float32)
        y_train = y_train.astype(np.float32)
        
        x_test = (x_test / 255.0).astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        x_train = np.reshape(x_train, [60000,28,28,1])
        x_test = np.reshape(x_test, [10000,28,28,1])
        if self.dataloader_args['da']:
            x_train, x_test = normalization(x_train, x_test)
        
        #on-hot
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        data_augmentation = tf.keras.Sequential([
                    preprocessing.Resizing(32,32)
                    ])

        full_size = len(x_train)
        test_size = len(x_test)
        
        train_size = int(1.0 * full_size)
        valid_size = int(0.5 * test_size)

        full_dataset = tf.data.Dataset.from_tensor_slices({'inputs': x_train, 'labels': y_train})
        full_dataset = full_dataset.shuffle(full_size)

        train_dataset = full_dataset.take(train_size)
        train_dataset = train_dataset.batch(self.dataloader_args['batch_size'], drop_remainder=True)
        # data augmentation
        if self.dataloader_args['da']:
            train_dataset = train_dataset.map(lambda x:{'inputs':data_augmentation(x['inputs']),'labels': x['labels']}, num_parallel_calls=16)
        train_dataset = train_dataset.prefetch(1)
        train_dataset = train_dataset.repeat(epochs)


        # valid_dataset = full_dataset.skip(train_size)
        # valid_dataset = valid_dataset.take(valid_size).repeat(epochs)
        # valid_dataset = valid_dataset.batch(self.dataloader_args['batch_size'])

        test_dataset = tf.data.Dataset.from_tensor_slices({'inputs': x_test, 'labels': y_test})
        test_dataset = test_dataset.shuffle(test_size)
        # valid_dataset = test_dataset.take(valid_size).batch(self.dataloader_args['batch_size']).repeat(epochs)
        # test_dataset = test_dataset.skip(valid_size).batch(self.dataloader_args['batch_size']).repeat(epochs)
        test_dataset = test_dataset.batch(self.dataloader_args['batch_size'], drop_remainder=True)
        if self.dataloader_args['da']:
            test_dataset = test_dataset.map(lambda x:{'inputs':data_augmentation(x['inputs']),'labels': x['labels']}, num_parallel_calls=16)
        test_dataset = test_dataset.repeat(epochs)
        valid_dataset = test_dataset

        return train_dataset, valid_dataset, test_dataset
    
if __name__ == "__main__":
    dataloader_args = edict({"batch_size": 128, "epochs": 50, "da": True})
    dataloader = MnistDataLoader(dataloader_args=dataloader_args)
    dataloader.load_datadict()