import tensorflow as tf

class SeqWiseBatchNorm(tf.keras.layers.Layer):

    def __init__(self, return_sequences = True):
        """
        param : return_sequences - return_sequences equivalent for the previous layer;
                        We need the previous layers' return_sequences to be True for seq_wise batchnorm
        """

        # For input-dependent initialisation
        super(SeqWiseBatchNorm, self).__init__()
        self.return_sequences = return_sequences

    def build(self, input_shape):
        # Rest of the initialisation: learnable params etc, now that we know the shape of the inputs
        self.Lambda = self.add_weight(shape= (), initializer = 'ones', trainable = True)
        self.beta = self.add_weight(shape= (), initializer = 'zeros', trainable = True)
        self.eps = 1e-3

    def call(self, input):
        # 'Forward' computation
        mean, var = tf.nn.moments(input, axes = [0, 1], keepdims = True)
        result = (input - mean)/tf.math.sqrt(var+self.eps)
        result = self.Lambda*result + self.beta
        if not return_sequences:
            result[:, -1, :]
        return result

class RowConv1D(tf.keras.layers.Layer):
    # TODO

    def __init__(self, None):
        super(RowConv1D, self).__init__()
        pass

    def build(self, input_shape):
        pass

    def call(self, input):
        pass