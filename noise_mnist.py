import os
import random
import numpy as np
import tensorflow as tf

from easydict import EasyDict as edict
import matplotlib.pyplot as plt

# modules
from dataloader import Cifar10DataLoader, MnistDataLoader
from dnn import DNN
from dcgan import DCNN
from unet import CUNet
from utiliz import display
  
train_loss_fn = tf.keras.losses.CategoricalCrossentropy()
mt_loss_fn = tf.keras.metrics.Mean()
test_loss_fn = tf.keras.losses.CategoricalCrossentropy()
mte_loss_fn = tf.keras.metrics.Mean()
opt_loss_fn = tf.keras.losses.categorical_crossentropy

train_metrics = tf.keras.metrics.CategoricalAccuracy()
test_metrics = tf.keras.metrics.CategoricalAccuracy()

GIAO_EPOCH = 1
GIAO_BATCH = 8


giao_optimizer = tf.keras.optimizers.Adam(1e-4)
giao_loss_fn = tf.keras.losses.MeanSquaredError()

# @tf.function(experimental_relax_shapes=True, experimental_compile=None)
def _train_step(model, optimizer, inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = train_loss_fn(labels, predictions)
        metrics = tf.reduce_mean(train_metrics(labels, predictions))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    mt_loss_fn.update_state(loss)
    
    return loss, metrics

def _test_step(model, inputs, labels):
    predictions = model(inputs)
    loss = test_loss_fn(labels, predictions)
    opt_loss = opt_loss_fn(labels, predictions)
    metrics = tf.reduce_mean(test_metrics(labels, predictions))
    mte_loss_fn.update_state(loss)
    
    return loss, metrics, opt_loss

# @tf.function(experimental_relax_shapes=True, experimental_compile=None)
def _giao_train_step(gnet, model, noise, fake_label, train_data, test_data):
    # noise = tf.random.normal([128, 100])
    with tf.GradientTape() as g_tape:
        
        # fake test loss
        fake_test_sample, _ = gnet(noise)
        fake_predictions = model(fake_test_sample)
        fake_test_loss = opt_loss_fn(fake_label, fake_predictions)
        
        # similarity loss
        l1_loss = tf.reduce_mean(tf.abs(train_data-fake_test_sample))
        
        #test loss
        test_predictions = model(test_data["inputs"])
        test_loss = opt_loss_fn(test_data["labels"], test_predictions)
      
        ftl = tf.reduce_mean(fake_test_loss)
        tl = tf.reduce_mean(test_loss)
        
        g_loss = tf.abs(ftl - tl) + l1_loss
        gradients = g_tape.gradient(g_loss, gnet.model.trainable_variables)
        giao_optimizer.apply_gradients(zip(gradients, gnet.model.trainable_variables))
    return fake_test_sample, l1_loss, g_loss, ftl, tl

    
def obtain_da_model_opts(gnet, model_args, fake_label, noise,
                         dataloader, iter_train, iter_test, 
                         sample_gap=3, epochs=3, giao_step=0):
    
    model = DNN(units=model_args.units, activations=model_args.activations)
    optimizer = tf.keras.optimizers.SGD(0.1)
    
    records = edict({'epoch':[],'train_loss':[],'test_loss':[],'train_metric':[],'test_metric':[]})
    for e in range(epochs):
        mt_loss_fn.reset_states()
        train_metrics.reset_states()
        mte_loss_fn.reset_states()
        test_metrics.reset_states()
        for step in range(dataloader.info.train_step):
            train_data = iter_train.get_next()
            train_loss, acc = _train_step(model=model,optimizer = optimizer, inputs=train_data["inputs"], labels=train_data["labels"])
            if (e*dataloader.info.train_step + step)%sample_gap == 0:
                test_data =  iter_test.get_next()
                fake_test_sample, geo_l1, g_loss, ftl, tl = _giao_train_step(gnet=gnet,
                                                                        fake_label=fake_label, noise=noise,
                                                                        model=model, train_data=train_data["inputs"], test_data=test_data)
                save_name = "{}_{}_{}".format(giao_step, e*dataloader.info.train_step+step, g_loss)
                if (e*dataloader.info.train_step + step)%(sample_gap*20) == 0:
                    print("FTL: {}, test loss:{}, geo loss:{}, fake label: {}".format(ftl, tl, geo_l1 ,tf.argmax(fake_label[32])))
                    display([train_data["inputs"][0], fake_test_sample[32], test_data["inputs"][0]], save_name=save_name)
                    
        test_loss, test_acc, _ = _test_step(model=model, inputs=test_data["inputs"], labels=test_data["labels"])
        records.epoch        += [e]
        records.train_loss   += [mt_loss_fn.result().numpy()]
        records.train_metric += [train_metrics.result().numpy()]
        records.test_loss    += [mte_loss_fn.result().numpy()]
        records.test_metric  += [test_metrics.result().numpy()]
        log = ""
        for k,v in records.items():
            log += "{}: {} ".format(k,v[-1])
        print(log)
        
def giao(gnet, model_args, dataloader, iter_train, iter_test):    
    #GIAO
    fake_label = iter_test.get_next()["labels"]
    noise = tf.random.normal([128, 32, 32, 1])
    for t in range(1000):
        # Train on Dasamples
        obtain_da_model_opts(gnet=gnet, model_args=model_args, fake_label=fake_label, noise=noise,
                            iter_train=iter_train, iter_test=iter_test, dataloader=dataloader,
                            sample_gap=5, epochs=1, giao_step=t)
    
def main():
    # dataset
    dataloader_args = edict({"batch_size": 128, "epochs": 20, "da": True})
    dataloader = MnistDataLoader(dataloader_args=dataloader_args)
    train_dataset, valid_dataset, test_dataset = dataloader.load_dataset()

    #Envioroment model
    model_args = edict({"units":[128,64,32,10], "activations":["relu","relu","relu","softmax"]})
    
    #Generator
    # gnet = DCNN()
    gnet = CUNet(input_shape=[32, 32, 1])
    
    iter_train = iter(train_dataset)
    iter_test = iter(test_dataset)
    
    giao(gnet=gnet, model_args=model_args, 
         dataloader=dataloader, iter_train=iter_train, iter_test=iter_test)

if __name__== "__main__":
    main()
