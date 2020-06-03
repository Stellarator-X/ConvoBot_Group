import numpy as np 
from tensorflow.keras import backend as K 
from tensorflow.keras.layers import Input, Dense, Activation, Lambda, GRU, Bidirectional, Conv1D, Conv2D, TimeDistributed
from tensorflow.keras.models import Sequential, Model
from ds_utils import SeqWiseBatchNorm

class DSModel():

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_len = None
    
    def build():
        sefl.model = Sequential([
            Conv2D(filters = 16, kernel_size=(3, 3), strides = 4 ,input_shape = self.input_shape),
            Conv2D(filters = 16, kernel_size = (3, 3), strides = 3),
            Conv1D(filters = 32, kernel_size=15, strides = 2),

            Bidirectional(GRU(units = 128, return_sequences=True)),
            SeqWiseBatchNorm(),
            Bidirectional(GRU(units = 128, return_sequences=True)),
            SeqWiseBatchNorm(),
            Bidirectional(GRU(units = 128, return_sequences=True)),
            SeqWiseBatchNorm(),
            Bidirectional(GRU(units = 128, return_sequences=True)),
            SeqWiseBatchNorm(),
            Bidirectional(GRU(units = 128, return_sequences=True)),
            SeqWiseBatchNorm(),
            Bidirectional(GRU(units = 128, return_sequences=True)),
            SeqWiseBatchNorm(),
            Bidirectional(GRU(units = 128, return_sequences=True)),
            SeqWiseBatchNorm(return_sequences= True),

            TimeDistributed(Dense(units = ALPHABET_LENGTH, activation=softmax))
            
        ], name = "DeepSpeech2")


    def ctc_find_eos(y_true, y_pred):
        # From StackOverflow : TODO : personal tweaks
        #convert y_pred from one-hot to label indices
        y_pred_ind = K.argmax(y_pred, axis=-1)

        #to make sure y_pred has one end_of_sentence (to avoid errors)
        y_pred_end = K.concatenate([
                                    y_pred_ind[:,:-1], 
                                    eos_index * K.ones_like(y_pred_ind[:,-1:])
                                ], axis = 1)

        #to make sure the first occurrence of the char is more important than subsequent ones
        occurrence_weights = K.arange(start = max_length, stop=0, dtype=K.floatx())

        #is eos?
        is_eos_true = K.cast_to_floatx(K.equal(y_true, eos_index))
        is_eos_pred = K.cast_to_floatx(K.equal(y_pred_end, eos_index))

        #lengths
        true_lengths = 1 + K.argmax(occurrence_weights * is_eos_true, axis=1)
        pred_lengths = 1 + K.argmax(occurrence_weights * is_eos_pred, axis=1)

        #reshape
        true_lengths = K.reshape(true_lengths, (-1,1))
        pred_lengths = K.reshape(pred_lengths, (-1,1))

        return K.ctc_batch_cost(y_true, y_pred, pred_lengths, true_lengths)

    def summary(self):
        self.model.summary()

    def compile(self, )
        self.model.compile(loss = ctc_find_eos, optimizer = 'adam', metrics = ['accuracy'])

    def fit(self, **kwargs):
        self.model.fit(**kwargs)