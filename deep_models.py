from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM


def model_1(input_dim, output_dim):
    model = Sequential()
    # model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(LSTM(128, input_shape=input_dim, return_sequences=True))
    # model.add(LSTM(128, input_shape=(maxlen, len(chars)), activation='relu', return_sequences=True, dropout=0.2))
    model.add(LSTM(128, input_shape=input_dim))
    # model.add(LSTM(128, activation='relu', dropout=0.2))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_1'


# summery of result for this model
def model_2(input_dim, output_dim):
    model = Sequential()
    # model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(LSTM(128, input_shape=input_dim, return_sequences=True, dropout=0.3, recurrent_dropout=0.1))
    model.add(LSTM(128, input_shape=input_dim, return_sequences=False, dropout=0.3, recurrent_dropout=0.1))
    # model.add(LSTM(128, activation='relu', dropout=0.2))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_2'


def model_3(input_dim, output_dim):
    model = Sequential()
    # model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(LSTM(256, input_shape=input_dim, return_sequences=True, dropout=0.4, recurrent_dropout=0.2))
    model.add(LSTM(256, input_shape=input_dim, return_sequences=False, dropout=0.4, recurrent_dropout=0.2))
    # model.add(LSTM(128, activation='relu', dropout=0.2))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model, 'model_3'

