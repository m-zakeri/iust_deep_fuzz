"""
PDF OBJ 2
Train with generator for large datasets

"""

from __future__ import print_function

import keras
from docutils.nodes import paragraph
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, LambdaCallback
from keras.utils import plot_model
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

import datetime
import pdf_object_preprocess as preprocess


def define_model(input_dim, output_dim):
    """build the model: a single LSTM layer # we need to deep it"""
    print('Build model...')
    model = Sequential()
    # model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(LSTM(128, input_shape=input_dim, return_sequences=True))
    # model.add(LSTM(128, input_shape=(maxlen, len(chars)),
    #                activation='relu', return_sequences=True, dropout=0.2))
    model.add(LSTM(128, input_shape=input_dim))
    # model.add(LSTM(128, activation='relu', dropout=0.2))
    model.add(Dense(output_dim))
    model.add(Activation('softmax'))
    return model


def load_all_samples(path, maxlen, step):
    text = preprocess.load_from_file(path)
    print('Corpus length:', len(text))
    chars = sorted(list(set(text)))
    print('Total chars:', len(chars))
    # print(chars)

    # Building dictionary index
    print('Building dictionary index ...')
    char_indices = dict((c, i) for i, c in enumerate(chars))
    # print(char_indices)
    indices_char = dict((i, c) for i, c in enumerate(chars))
    # print(indices_char)

    # cut the text in semi-redundant sequences of maxlen characters
    sentences = []  # list of all sentence as input
    next_chars = []  # list of all next chars as labels
    for i in range(0, len(text) - maxlen, step):  # arg2 why this?
        sentences.append(text[i: i + maxlen])
        # print(sentences)
        next_chars.append(text[i + maxlen])
        # print(next_chars)
    print('Number of semi sequences:', len(sentences))
    return sentences, next_chars, chars, char_indices, indices_char, text


def data_generator(samples, batch_size, maxlen, chars, char_indices, next_chars):
    j = 0
    print('Vectorization...')
    x = np.zeros((batch_size, maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((batch_size, len(chars)), dtype=np.bool)
    while True:
        j = random.randint(0, len(samples)-1)
        for i, one_sample in enumerate(samples[j: j + batch_size]):
            for t, char in enumerate(one_sample):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1
        yield x, y
        # j += batch_size


def train():
    # define some hyper-parameter
    maxlen = 85  # Good idea?: use ave_object_len or StdDev to determine this hyper-parameter
    step = 1  # should set to 1 for best result?
    batch_size = 128
    epoch_list = [2, 10, 20, 30, 40, 50]
    ts_path = './dataset/pdfs/small_size_dataset/' \
              '06_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_trainset_47711_shuffle_ii.txt'
    vs_path = './dataset/pdfs/small_size_dataset/' \
              '06_object_id__object_len_485080_iqr_cleaned_with_2_column_477104_validationset_11927_shuffle_ii.txt'
    ts_sentences, ts_next_chars, ts_chars, ts_char_indices, ts_indices_char, ts_text = load_all_samples(
        ts_path, maxlen, step)

    vs_sentences, vs_next_chars, vs_chars, vs_char_indices, vs_indices_char, vs_text = load_all_samples(
        vs_path, maxlen, step)
    # input()

    model = define_model((maxlen, len(ts_chars)), len(ts_chars))
    optimizer = RMSprop(lr=0.02)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    ts_data_generator = data_generator(ts_sentences, batch_size, maxlen, ts_chars, ts_char_indices, ts_next_chars)
    vs_data_generator = data_generator(vs_sentences, batch_size, maxlen, ts_chars, ts_char_indices, vs_next_chars)

    # set callbacks
    dt = datetime.datetime.now().strftime('date_%Y%m%d_%H%M%S_')
    model_checkpoint_filepath = './model_checkpoint/'+dt+'epoch_{epoch:02d}_{val_loss:.4f}.h5'
    model_chekpoint = ModelCheckpoint(model_checkpoint_filepath, verbose=1)

    model_early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=0, verbose=0, mode='auto')

    model_tensorboard = TensorBoard(log_dir='./logs_tensorboard', histogram_freq=0, batch_size=batch_size,
                                    write_graph=True,write_grads=False, write_images=True, embeddings_freq=0,
                                    embeddings_layer_names=None, embeddings_metadata=None)
    model_csv_logger_filepath = './logs_csv/training_with_epoch_02.csv'
    model_csv_logger = CSVLogger(model_csv_logger_filepath, separator=',', append=False)

    generate_and_fuzz_new_samples_callback = LambdaCallback(on_epoch_begin=None,
                                                            on_epoch_end=generate_and_fuzz_new_samples,
                                                            on_batch_begin=None, on_batch_end=None,
                                                            on_train_begin=None, on_train_end=None)

    print('Start training...')
    model.fit_generator(generator=ts_data_generator,
                        steps_per_epoch=len(ts_sentences) // batch_size,
                        validation_data=vs_data_generator,
                        validation_steps=len(vs_sentences) // batch_size,
                        epochs=epoch_list[0],
                        callbacks=[model_chekpoint, model_early_stopping, model_tensorboard, model_csv_logger])

    # model.fit(x, y, batch_size=128, epochs= epoch_list[0], validation_split=0.2)  # why epochs=?
    # save(model,  epoch_list[0])
    # del model
    # model = load_model('./modelh5/lstm_text_generation_pdf_objs_1_20180214_235713_epochs10.h5')


def generate_and_fuzz_new_samples(epoch, logs):
    """ sampling the model and generate new object """
    # fuzzing hyperparameters
    # diversities = [i*0.10 for i in range(1,20,2)]
    diversities = [0.2, 0.5, 1.0, 1.2, 1.5, 1.8]
    generated_obj_max_number = 5
    generated_obj_max_allowed_len = 500
    t_fuzz = 0.9
    p_t = 0.9  # 0.9 for format fuzzing and 0.5 or letter for data fuzzing. Now format fuzzing
    # end of fuzzing parameters

    list_of_objects = preprocess.get_list_of_object(ts_text)
    list_of_objects_with_maxlen = []
    for o in list_of_objects:
        if len(o) > maxlen:
            list_of_objects_with_maxlen.append(o)

    for diversity in diversities:
        obj_index = random.randint(0, len(list_of_objects_with_maxlen) - 1)
        generated_obj_counter = 0
        generated_obj_len_index = 0

        stop_condition = False
        print()
        print('-- Diversity:', diversity)

        # generated = ''
        obj_prefix = str(list_of_objects_with_maxlen[obj_index])[0: maxlen]  # len(sentence) equals 100 here

        generated = obj_prefix
        prob_vals = '100\n' * maxlen
        learnt_grammar = obj_prefix

        print('--- Generating ts_text with seed:\n "' + obj_prefix + '"')
        sys.stdout.write(generated)

        if generated.endswith('endobj'):
            generated_obj_counter += 1

        if generated_obj_counter > generated_obj_max_number:
            stop_condition = True

        while not stop_condition:
            x_pred = np.zeros((1, maxlen, len(ts_chars)))
            for t, char in enumerate(obj_prefix):
                x_pred[0, t, ts_char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index, prob, preds2 = sample(preds, diversity)
            p_fuzz = random.random()
            if p_fuzz > t_fuzz and preds2[next_index] > p_t:
                next_index = np.argmin(preds2)
                print('FUZZ DONE!')

            next_char = ts_indices_char[next_index]

            # print()
            # print(preds2)
            # print(np.argmax(preds))
            # print(preds[np.argmax(preds)])

            # print(prob)
            # print(np.argmax(prob))
            # print('====>',next_index)
            # print(prob[0, next_index])

            # prob_vals += str(preds2[next_index]) + '\n'
            # if preds2[next_index] > 0.9980:
            #     learnt_grammar += next_char
            # else:
            #     learnt_grammar += '.'
            # input()

            obj_prefix = obj_prefix[1:] + next_char
            generated += next_char
            generated_obj_len_index += 1

            if generated.endswith('endobj'):
                generated_obj_counter += 1
                generated_obj_len_index = 0
            elif generated_obj_len_index > generated_obj_max_allowed_len:
                generated += '\nendobj\n'
                generated_obj_counter += 1
                generated_obj_len_index = 0

            if generated_obj_counter > generated_obj_max_number:
                stop_condition = True

            sys.stdout.write(next_char)
            sys.stdout.flush()

        # save generated ts_text to file inside program
        dt = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S_')
        dir_name = './generated_results/pdfobjs_new/'
        file_name = 'gen_objs' + dt + 'epochs' + repr(epoch_list[0]) + '_div' \
                    + repr(diversity) + '_step' + repr(step) + '.txt'
        preprocess.save_to_file(dir_name + file_name, generated)
        # poc.save_to_file(dir_name + file_name + 'probabilities.txt', prob_vals)
        # poc.save_to_file(dir_name + file_name + 'learntgrammar.txt',learnt_grammar)


# lower temperature will cause the model to make more likely,
# but also more boring and conservative predictions.
def sample(preds, temperature=1.0):
    """helper function to sample an index from a probability array"""
    # print('raw predictions = ', preds)
    preds = np.asarray(preds).astype('float64')

    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    """sampling:"""
    probas = np.random.multinomial(1, preds, 1)
    # print()
    # print('sanitize predictions = ', preds)
    return np.argmax(probas), probas, preds


def no_sample():
    pass


def sample_space():
    pass


def save_model_plot(model, epochs):
    """ save the model architecture plot """
    dt = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S_')
    # plot the model
    plot_model(model, to_file='./modelpic/date_' + dt + 'epochs_' + str(epochs) + '.png',
               show_shapes=True, show_layer_names=True)


def main(argv):
    """ the main function """
    train()


if __name__ == "__main__":
    main(sys.argv)
