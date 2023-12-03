import os
import random
import numpy as np
import tensorflow as tf

from easydict import EasyDict as edict
import matplotlib.pyplot as plt

# modules
from dataloader import Cifar10DataLoader, MnistDataLoader
from dnn import DNN
from unet import CUNet
from diffusion import DiffusionUnet
from utiliz import display
  
train_loss_fn = tf.keras.losses.CategoricalCrossentropy()
mt_loss_fn = tf.keras.metrics.Mean()
test_loss_fn = tf.keras.losses.CategoricalCrossentropy()
mte_loss_fn = tf.keras.metrics.Mean()
opt_loss_fn = tf.keras.losses.CategoricalCrossentropy()

train_metrics = tf.keras.metrics.CategoricalAccuracy()
test_metrics = tf.keras.metrics.CategoricalAccuracy()
optimizer = tf.keras.optimizers.SGD(0.1)

GIAO_EPOCH = 1
GIAO_BATCH = 8


giao_optimizer = tf.keras.optimizers.Adam(1e-4)
giao_loss_fn = tf.keras.losses.MeanSquaredError()

@tf.function(experimental_relax_shapes=True, experimental_compile=None)
def _train_step(model, inputs, labels):
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

def collect_model_operator(model_args, variables, loss):
    weights = [w.numpy() for w in variables]
    opt = DNN(units=model_args.units, 
            activations=model_args.activations,
            init_value=weights)
    return opt, loss


def obtain_model_opts(model, model_args, 
                      dataloader, iter_train, test_data, 
                      sample_start=30, sample_gap=20):
    model_opt = []
    opt_label = []
    records = edict({'epoch':[],'train_loss':[],'test_loss':[],'train_metric':[],'test_metric':[]})
    for e in range(dataloader.info.epochs):
        mt_loss_fn.reset_states()
        train_metrics.reset_states()
        mte_loss_fn.reset_states()
        test_metrics.reset_states()
        for step in range(dataloader.info.train_step):
            data = iter_train.get_next()
            train_loss, acc = _train_step(model=model, inputs=data["inputs"], labels=data["labels"])
            if (e*dataloader.info.train_step + step)%sample_gap ==0:
                if e >= sample_start:
                    test_loss, test_acc, opt_loss = _test_step(model=model, inputs=test_data["inputs"], labels=test_data["labels"])
                    opt, label = collect_model_operator(model_args, model.trainable_variables, opt_loss)
                    model_opt.append(opt)
                    opt_label.append(label)
                    
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
        [reg(test_data["inputs"]) for reg in model_opt] # init model reg
    return model_opt, opt_label

def init_model_opt(raw_model_opt, test_data):
    
    def opt_test_step(opt, inputs, labels):
        predictions = opt(inputs)
        loss = test_loss_fn(labels, predictions)
        metrics = tf.reduce_mean(test_metrics(labels, predictions))
        mte_loss_fn.update_state(loss)
        return loss, metrics

    for idx in range(len(raw_model_opt)):
        mte_loss_fn.reset_states()
        test_metrics.reset_states()
        for step in range(1):
            data = test_data
            test_loss, test_acc = opt_test_step(opt=raw_model_opt[idx], inputs=data["inputs"], labels=data["labels"])
        print("Init: opt_id:{}, Test loss:{}, Test acc:{}".format(idx,
                                                        mte_loss_fn.result().numpy(),
                                                        test_metrics.result().numpy()))
            
def hard_save_model_opt(online_model_opt, path="./model_opt", test_data=None):
    init_model_opt(online_model_opt, test_data)
    for idx in range(len(online_model_opt)):
        mpath = os.path.join(path, "opt_{}".format(idx))
        online_model_opt[idx].save(mpath, overwrite=True, save_format='tf')

def load_model_opt(path="./model_opt", test_data=None):
    offline_model_opt = []
    model_opt_list = os.listdir(path=path)
    for idx in range(len(model_opt_list)):
        mpath = os.path.join(path,  "opt_{}".format(idx))
        offline_model_opt.append(tf.keras.models.load_model(mpath))
    init_model_opt(offline_model_opt, test_data)
    return offline_model_opt

# @tf.function(experimental_relax_shapes=True, experimental_compile=None)
def _opt_train_step(unet, regs, train_inputs, train_labels, labels):
    losses = []
    
    with tf.GradientTape() as tape:
        pseudo_inputs, train_latent_phase = unet(train_inputs)
        _, pseudo_latent_phase= unet(pseudo_inputs, training=False)
      
        for idx in range(len(regs)):
            predictions = regs[idx](pseudo_inputs)
            l1_loss = tf.reduce_mean(tf.abs(pseudo_inputs - train_inputs)) * 100
            phase_loss = tf.reduce_mean(tf.square(pseudo_latent_phase - train_latent_phase)) * 0
            reg_loss = opt_loss_fn(train_labels, predictions)
            giao_loss =  tf.abs(labels[idx]-reg_loss) + phase_loss + l1_loss
            losses.append(giao_loss)
        loss = tf.reduce_mean(losses)
        grad = tape.gradient(loss, unet.model.trainable_variables)
        
        
    giao_optimizer.apply_gradients(zip(grad, unet.model.trainable_variables))
    return loss, l1_loss, phase_loss, pseudo_inputs

def obtain_da_model_opts(unet, model, model_args,
                         dataloader, iter_train, test_data, 
                         sample_start=0, sample_gap=3, epochs=3, giao_step=0):
    model_opt = []
    opt_label = []
    records = edict({'epoch':[],'train_loss':[],'test_loss':[],'train_metric':[],'test_metric':[]})
    for e in range(epochs):
        mt_loss_fn.reset_states()
        train_metrics.reset_states()
        mte_loss_fn.reset_states()
        test_metrics.reset_states()
        for step in range(dataloader.info.train_step):
            data = iter_train.get_next()
            with tf.GradientTape() as g_tape:
                da_data_inputs, _ = unet(data["inputs"])
                train_loss, acc = _train_step(model=model, inputs=da_data_inputs, labels=data["labels"])
                if (e*dataloader.info.train_step + step)%sample_gap == 0:
                    if e >= sample_start:
                        test_loss, test_acc, opt_loss = _test_step(model=model, inputs=test_data["inputs"], labels=test_data["labels"])
                        giao_loss = tf.reduce_mean(tf.square(test_loss-train_loss))
                        g_grad = g_tape.gradient(giao_loss, unet.model.trainable_variables)
                        giao_optimizer.apply_gradients(zip(g_grad, unet.model.trainable_variables))
                        
                        save_name = "{}_{}_{}_{}".format(giao_step, e, step, giao_loss) 
                        if (e*dataloader.info.train_step + step)%(sample_gap*10) == 0:
                            display([data["inputs"].numpy()[0], da_data_inputs.numpy()[0], test_data["inputs"][0]], save_name=save_name)
                    
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
        
def giao(unet, model,  model_args, dataloader, iter_train, test_data):
    
    # init training
    model_opt, opt_label = obtain_model_opts(model=model, model_args=model_args, 
                                             iter_train=iter_train, test_data=test_data,
                                             dataloader=dataloader, sample_start=0, sample_gap=100) 
    print(len(model_opt))
    
    #GIAO
    for t in range(100):
        #Generator training
        if t == 0:
            TRAIN_STEP = 1000
            for j in range(TRAIN_STEP):
                train_data = iter_train.get_next()
                idx = random.sample(range(len(model_opt)), GIAO_BATCH)
                labels = [opt_label[i] for i in idx]
                regs = [model_opt[i] for i in idx]
                giao_train_loss, l1_loss, phase_loss, pseudo_inputs = _opt_train_step(unet, regs, train_data["inputs"], test_data["labels"], labels)
                if j % 100 == 0:
                    print("Step:{} GIAO Loss:{}, L1 Loss:{}, Phase Loss:{}".format(t*TRAIN_STEP + j, 
                                                                                giao_train_loss-l1_loss-phase_loss, 
                                                                                l1_loss, phase_loss))
                    save_name = "{}_{}_{}".format(t,j,giao_train_loss) 
                    display([train_data["inputs"].numpy()[0], pseudo_inputs.numpy()[0], test_data["inputs"][0]], save_name=save_name)
                
        # Train on Dasamples
        obtain_da_model_opts(unet=unet, model=model, model_args=model_args, 
                                                    iter_train=iter_train, test_data=test_data, dataloader=dataloader,
                                                    sample_start=0, sample_gap=10, epochs=3, giao_step=t)
        print(len(model_opt))
    
def main():
    # dataset
    dataloader_args = edict({"batch_size": 128, "epochs": 50, "da": False})
    dataloader = Cifar10DataLoader(dataloader_args=dataloader_args)
    train_dataset, valid_dataset, test_dataset = dataloader.load_dataset()

    #Envioroment
    model_args = edict({"units":[128,64,32,10], "activations":["relu","relu","relu","softmax"]})
    model = DNN(units=model_args.units, activations=model_args.activations)
    
    #Generator
    # unet = UNet(input_shape=[32, 32, 3])
    unet = CUNet(input_shape=[32, 32, 3])
    # unet = DiffusionUnet()
    
    iter_train = iter(train_dataset)
    iter_test = iter(test_dataset)
    
    test_data =  iter_test.get_next()
    
    giao(unet=unet, model=model, model_args=model_args, 
         dataloader=dataloader, iter_train=iter_train, test_data=test_data)

if __name__== "__main__":
    main()
