from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM


# Keras LSTM text generation example model (simplest model)
# summery of result for model_0 (Not deep model):
#
#
def model_0(input_dim, output_dim):
    """
    Total params: 127,584
    Trainable params: 127,584
    Non-trainable params: 0

    :param input_dim:
    :param output_dim:
    :return:
    """
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=input_dim))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_0'


# summery of result for model_1 (deep 2):
#
#
def model_1(input_dim, output_dim):
    """
    Total params: 259,168
    Trainable params: 259,168
    Non-trainable params: 0

    :param input_dim:
    :param output_dim:
    :return:
    """
    model = Sequential()
    # model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(LSTM(128, input_shape=input_dim, return_sequences=True))
    # model.add(LSTM(128, input_shape=(maxlen, len(chars)), activation='relu', return_sequences=True, dropout=0.2))
    model.add(LSTM(128, input_shape=input_dim))
    # model.add(LSTM(128, activation='relu', dropout=0.2))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_1'


# Summery of result for model_2 (deep 2):
# Test done!
# Bad model
# Not good:-(. The model loss stop on 1.88 after 18 epoch run on cpu (deepubuntu)
# <><><><><> Model compile config: <><><><><><>
# optimizer = RMSprop(lr=0.01)  # [0.01, 0.02, 0.05, 0.1]
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
def model_2(input_dim, output_dim):
    """
    Total params: 259,168
    Trainable params: 259,168
    Non-trainable params: 0

    :param input_dim:
    :param output_dim:
    :return:
    """
    model = Sequential()
    # model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(LSTM(128, input_shape=input_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))
    model.add(LSTM(128, input_shape=input_dim, return_sequences=False, dropout=0.2, recurrent_dropout=0.1))
    # model.add(LSTM(128, activation='relu', dropout=0.2))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_2'


# Summery of result for this model:
def model_3(input_dim, output_dim):
    """
    Total params: 911,456
    Trainable params: 911,456
    Non-trainable params: 0

    :param input_dim:
    :param output_dim:
    :return:
    """
    model = Sequential()
    # model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(LSTM(256, input_shape=input_dim, return_sequences=True, dropout=0.4, recurrent_dropout=0.2))
    model.add(LSTM(256, input_shape=input_dim, return_sequences=False, dropout=0.4, recurrent_dropout=0.2))
    # model.add(LSTM(128, activation='relu', dropout=0.2))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_3'


# Summery of result for this model:
# Try 1:
# * When learning rate is 0.01 and batch select sequentially the loss stuck on 2.3454 after about 6 epochs.
# This is very  bad -:(
#
# Try 2:
# So we changed lr and also batch selection:
# Result for: lr=0.001, batch select randomly! (Shuffle), step=3, batch_size=128, maxlen=50
# * Pretty good with above config on small dataset
#
# Try 3:
# Train on large dataset (prefix choose from the train-set not test-set) on generation phase:
# Bad result after 6 epoch (the loss decrease suddenly.
#
# Try 4:
# bach_size=256, lr=0.001, step=1, maxlen=50
#
# Totally not good model
def model_4(input_dim, output_dim):
    """
        Total corpus length: 11,530,647
        Total corpus chars: 96
        Building dictionary index ...
        Get model summary ...
        model_4  summary ...
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        lstm_1 (LSTM)                (None, 85, 256)           361472
        _________________________________________________________________
        dropout_1 (Dropout)          (None, 85, 256)           0
        _________________________________________________________________
        lstm_2 (LSTM)                (None, 256)               525312
        _________________________________________________________________
        dropout_2 (Dropout)          (None, 256)               0
        _________________________________________________________________
        dense_1 (Dense)              (None, 96)                24672
        _________________________________________________________________
        activation_1 (Activation)    (None, 96)                0
        =================================================================
        Total params: 911,456
        Trainable params: 911,456
        Non-trainable params: 0
        _________________________________________________________________
        model_4  count_params ...
        911456

        :param input_dim:
        :param output_dim:
        :return:
        """
    model = Sequential()
    model.add(LSTM(256, input_shape=input_dim, return_sequences=True, recurrent_dropout=0.1))
    model.add(Dropout(0.2))
    model.add(LSTM(256, input_shape=input_dim, return_sequences=False, recurrent_dropout=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_4'


# Summery of result for this model:
# Try 1:
# batch_size=128, lr=0.001
#
#
#
#
def model_5(input_dim, output_dim):
    """
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    lstm_1 (LSTM)                (None, 50, 64)            41216
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 50, 64)            0
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 50, 64)            33024
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 50, 64)            0
    _________________________________________________________________
    lstm_3 (LSTM)                (None, 64)                33024
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 64)                0
    _________________________________________________________________
    dense_1 (Dense)              (None, 96)                6240
    _________________________________________________________________
    activation_1 (Activation)    (None, 96)                0
    =================================================================
    Total params: 113,504
    Trainable params: 113,504
    Non-trainable params: 0
    _________________________________________________________________
    model_5  count_params ...
    113504

    :param input_dim:
    :param output_dim:
    :return:
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_dim))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True, input_shape=input_dim))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_5'


# Summery of result for this model:
# Try 3:
# batch_size=128, lr=0.001
#
#
#
#
def model_6(input_dim, output_dim):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_dim, return_sequences=True, recurrent_dropout=0.1))
    model.add(Dropout(0.3))
    model.add(LSTM(128, input_shape=input_dim, return_sequences=False, recurrent_dropout=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_6'


# Summery of result for this model:
# Try 3:
# batch_size=128, lr=0.001
# With step 1 and neuron size 128 was very bad. Set step=3 and neuron size=256 and step=3
#
#
#
def model_7(input_dim, output_dim):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_dim, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(128, input_shape=input_dim, return_sequences=False))
    # model.add(Dropout(0.2))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_7'