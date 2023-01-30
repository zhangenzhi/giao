import os
import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict

# modules
from utiliz import check_mkdir
from dataloader import Cifar10DataLoader
from dnn import DNN

# dataloader
dataloader_args = edict({"batch_size": 128, "epochs": 100})
dataloader = Cifar10DataLoader(dataloader_args=dataloader_args)
source_train_ds,source_test_ds,target_train_ds,target_test_ds = dataloader.load_dataset()


# base model
model_args = edict({"units":[128,64,32,5], "activations":["relu","relu","relu","softmax"]})
model = DNN(units=model_args.units, activations=model_args.activations)


# metrics
train_loss_fn = tf.keras.losses.CategoricalCrossentropy()
mt_loss_fn = tf.keras.metrics.Mean()
test_loss_fn = tf.keras.losses.CategoricalCrossentropy()
mte_loss_fn = tf.keras.metrics.Mean()

train_metrics = tf.keras.metrics.CategoricalAccuracy()
test_metrics = tf.keras.metrics.CategoricalAccuracy()
max_metrics = tf.keras.metrics.Mean()
optimizer = tf.keras.optimizers.SGD(0.1)


# @tf.function(experimental_relax_shapes=True, experimental_compile=None)
def _train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = train_loss_fn(labels, predictions)
        metrics = tf.reduce_mean(train_metrics(labels, predictions))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    mt_loss_fn.update_state(loss)
    
    return loss, metrics

def _test_step(inputs, labels):
    predictions = model(inputs)
    loss = test_loss_fn(labels, predictions)
    metrics = tf.reduce_mean(test_metrics(labels, predictions))
    mte_loss_fn.update_state(loss)
    
    return loss, metrics

def copy_weights(variables):
    weights = [w.numpy() for w in variables]
    copied_model = DNN(units=model_args.units, 
                            activations=model_args.activations,
                            init_value=weights)
    return copied_model
    
def train_source_models(sample_gap=20):
    source_iter_train = iter(source_train_ds)
    source_iter_test = iter(source_test_ds)
    model_buffer = []
    for e in range(dataloader.source_info.epochs):
        mt_loss_fn.reset_states()
        train_metrics.reset_states()
        mte_loss_fn.reset_states()
        test_metrics.reset_states()
        for step in range(dataloader.source_info.train_step):
            data = source_iter_train.get_next()
            train_loss, acc = _train_step(inputs=data["inputs"], labels=data["labels"])
            if (e*dataloader.source_info.train_step + step)%sample_gap ==0:
                model_buffer.append(copy_weights(model.trainable_variables))
        for step in range(dataloader.source_info.test_step):
            data = source_iter_test.get_next()
            test_loss, test_acc = _test_step(inputs=data["inputs"], labels=data["labels"])
        print("Epoch:{}, Train loss: {}, Train acc: {}, Test loss:{}, Test acc:{}".format(e,
                                                                    mt_loss_fn.result().numpy(), 
                                                                    train_metrics.result().numpy(),
                                                                    mte_loss_fn.result().numpy(),
                                                                    test_metrics.result().numpy()))
    return model_buffer

def gmodel_test_step(gmodel, inputs, labels):
    predictions = gmodel(inputs)
    loss = test_loss_fn(labels, predictions)
    metrics = tf.reduce_mean(test_metrics(labels, predictions))
    mte_loss_fn.update_state(loss)
    return loss, metrics

def test_models_on_targets(model_buffer):
    source_iter_test = iter(source_test_ds)
    for idx in range(len(model_buffer)):
        mte_loss_fn.reset_states()
        test_metrics.reset_states()
        for step in range(1):
            data = source_iter_test.get_next()
            test_loss, test_acc = gmodel_test_step(gmodel=model_buffer[idx], inputs=data["inputs"], labels=data["labels"])
        print("M_id:{}, Test loss:{}, Test acc:{}".format(idx,
                                                        mte_loss_fn.result().numpy(),
                                                        test_metrics.result().numpy()))

def hard_save_gmodels(gmodels, path="./models"):
    check_mkdir(path=path)
    for idx in range(len(gmodels)):
        mpath = os.path.join(path, "gmodel_{}".format(idx))
        gmodels[idx].save(mpath, overwrite=True, save_format='tf')

def load_gmodels_hard(path="./models"):
    gmodels = []
    gmodel_list = os.listdir(path=path)
    for idx in range(len(gmodel_list)):
        mpath = os.path.join(path,  "gmodel_{}".format(idx))
        gmodels.append(tf.keras.models.load_model(mpath))
    return gmodels

def _gmodel_train_step(gmodels, inputs, labels, gopt):
    ggrads = []
    for m in gmodels:
        with tf.GradientTape() as tape:
            predictions = m(inputs)
            loss = train_loss_fn(labels, predictions)
            metrics = tf.reduce_mean(train_metrics(labels, predictions))
            grad = tape.gradient(loss, m.trainable_variables)
        ggrads.append(grad)
        mt_loss_fn.update_state(loss)
        
    mgrad = []
    for i in range(len(ggrads[0])):
        w = []
        for j in range(len(gmodels)):
            w.append(ggrads[j][i])
        mgrad.append(tf.reduce_sum(w, axis=0))
        
    for m in gmodels:
        gopt.apply_gradients(zip(mgrad, m.trainable_variables))
    
    return loss, metrics

def _gmodel_test_step(gmodels, inputs, labels):
    losses = []
    m_metrics = []
    for m in gmodels:
        predictions = m(inputs)
        loss = test_loss_fn(labels, predictions)
        losses.append(loss)
        metrics = tf.reduce_mean(test_metrics(labels, predictions))
        m_metrics.append(metrics)
        mte_loss_fn.update_state(loss)
    max_metrics.update_state(max(m_metrics))
    return losses, metrics

def L2L(model_buffer, n=24):
    target_iter_train = iter(target_train_ds)
    target_iter_test = iter(target_test_ds)
    data = target_iter_train.get_next()
    optimizer.lr = 0.01
    gopt = optimizer
    import random
    model_idx = random.sample(range(900, len(model_buffer)), n)
    gmodels = [model_buffer[idx] for idx in model_idx]
    
    for e in range(10000):
        mt_loss_fn.reset_states()
        train_metrics.reset_states()
        mte_loss_fn.reset_states()
        test_metrics.reset_states()
        max_metrics.reset_states()
        
        for step in range(1):
            train_loss, train_acc = _gmodel_train_step(gmodels=gmodels, 
                                                    inputs=data["inputs"], 
                                                    labels=data["labels"],
                                                    gopt=gopt)
        # for step in range()
        for step in range(10):
            te_data = target_iter_test.get_next()
            test_loss, test_acc = _gmodel_test_step(gmodels=gmodels, 
                                                    inputs=te_data["inputs"], 
                                                    labels=te_data["labels"]
                                                    )
        print("Epoch:{}, Train loss: {}, Train acc: {}, Test loss:{}, Test acc:{}, Max acc:{}".format(e,
                                                        mt_loss_fn.result().numpy(), 
                                                        train_metrics.result().numpy(),
                                                        mte_loss_fn.result().numpy(),
                                                        test_metrics.result().numpy(), max_metrics.result().numpy()))

def main():
    
    # generate base models
    # model_buffer = train_source_models(sample_gap=20) 
    
    # save/load to/from local
    # model_buffer = []
    # test_models_on_targets(model_buffer)
    # hard_save_gmodels(gmodels=model_buffer)
    
    model_buffer = load_gmodels_hard()
    test_models_on_targets(model_buffer)
    
    L2L(model_buffer = model_buffer)

if __name__== "__main__":
    main()
