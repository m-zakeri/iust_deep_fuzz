from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM


# Keras LSTM text generation example model (simplest model)
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


# summery of result for this model:
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


# summery of result for this model:
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


def model_3(input_dim, output_dim):
    pass

