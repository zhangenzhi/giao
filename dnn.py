import tensorflow as tf
from tensorflow import keras

class Linear(keras.layers.Layer):
    def __init__(self, units=32, seed=100000, init_value=None):

        super(Linear, self).__init__()
        self.seed = seed
        self.units = units
        self.init_value = init_value


    def build(self, input_shape):
        
        if self.init_value==None:
            w_init = tf.random_normal_initializer(seed=self.seed)(
                shape=(input_shape[-1], self.units), dtype="float32")
            b_init = tf.ones_initializer()(shape=(self.units,), dtype="float32")
        else:   
            w_init = self.init_value[0]
            b_init = self.init_value[1]

        self.w = tf.Variable(
            initial_value=w_init,
            trainable=True, name="w",
            constraint=lambda z: tf.clip_by_value(z, -10.0, 10.0)
        )
        self.b = tf.Variable(
            initial_value=b_init, trainable=True,
            name="b",
            constraint=lambda z: tf.clip_by_value(z, -10.0, 10.0)
        )

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.w) + self.b
        return outputs


class DNN(tf.keras.Model):
    def __init__(self,
                 units=[64, 32, 16, 1],
                 activations=['tanh', 'tanh', 'tanh', 'tanh'],
                 init_value=None,
                 seed=100000,
                 **kwargs
                 ):
        super(DNN, self).__init__()

        self.units = units
        self.activations = activations
        self.seed = seed
        
        self.exp = None
        
        self.flatten = tf.keras.layers.Flatten()
        self.fc_act = self._build_act()
        
        if init_value == None:
            self.fc_layers = self._build_fc()
        else:
            self.fc_layers = self._build_from_value(init_value=init_value)

    def _build_fc(self):
        layers = []
        for units in self.units:
            layers.append(Linear(units=units, seed=self.seed))
        return layers
    
    def _build_from_value(self, init_value):
        layers = []
        for i in range(len(self.units)):
            layers.append(Linear(units=self.units[i], seed=self.seed,
                                 init_value=(init_value[2*i], init_value[2*i+1])))
        return layers
            
    def _build_act(self):
        acts = []
        for act in self.activations:
            acts.append(tf.keras.layers.Activation(act))
        return acts

    def call(self, inputs):
        
        x = inputs
        x = self.flatten(x)
        feat_constrain = None
        
        for layer, act in zip(self.fc_layers, self.fc_act):
            x = layer(x)
            x = act(x)
            if feat_constrain == None:
                feat_constrain = x
            
        return x, feat_constrain