"""
PDF OBJ 1
A more complete version of this script is PDF OBJ 3.
Please refer to PDF OBJ 3 (lstm_text_generation_pdf_objs_3.py)

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
"""

from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

import datetime
import pdf_object_preprocess as poc


def save(model, epochs):
    """ save the model and the plot """
    dt = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S_')
    save_name = 'lstm_text_generation_pdf_objs_1' + dt + 'epochs' + str(epochs)
    # plot the model
    plot_model(model, to_file='./modelpic/' + save_name + '.png',
               show_shapes=True, show_layer_names=True)
    # Save model (requires HDF5 and h5py)
    model.save('./modelh5/' + save_name + '.h5')


def train():
    trainset_path = './trainset/pdfobjs.txt'
    trainset_path = './trainset/pdf_object_trainset_100_to_500_percent01.txt'
    text = poc.load_from_file(trainset_path)
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('Total chars:', len(chars))
    # print(chars)

    # Vectorization
    print('Building dictionary index ...')
    char_indices = dict((c, i) for i, c in enumerate(chars))
    # print(char_indices)
    indices_char = dict((i, c) for i, c in enumerate(chars))
    # print(indices_char)

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 50  # Good idea: use ave_object_len to determine this hyper-parameter
    step = 1  # should set to 1 for best result
    epochs = 10  # number of epochs for training
    sentences = []  # list of all sentence as input
    next_chars = []  # list of all next chars as labels
    for i in range(0, len(text) - maxlen, step):  # arg2 why this?
        sentences.append(text[i: i + maxlen])
        # print(sentences)
        next_chars.append(text[i + maxlen])
        # print(next_chars)

    print('semi sequences:', len(sentences))

    print('One-Hot vectorization...')
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)  # input x
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)  # output label y
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # build the model: a single LSTM layer # we need to deep it
    print('Build model...')
    model = Sequential()
    # model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(LSTM(128, input_shape=(maxlen, len(chars)), return_sequences=True))
    # model.add(LSTM(128, input_shape=(maxlen, len(chars)),
    #                activation='relu', return_sequences=True, dropout=0.2))
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    # model.add(LSTM(128, activation='relu', dropout=0.2))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    input()
    # sys.exit()

    model.fit(x, y, batch_size=128, epochs=epochs, validation_split=0.2)  # why epochs=?
    save(model, epochs)

    # del model
    # model = load_model('./modelh5/lstm_text_generation_pdf_objs_1_20180214_235713_epochs10.h5')

    """ sampling the model and generate new object """
    diversities = [0.2, 0.5, 1.0, 1.2, 1.5, 1.8]
    # diversities = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.2, 1.5, 1.7, 2]
    generated_obj_max_number = 5
    generated_obj_max_allowed_len = 500
    t_fuzz = 0.9
    p_t = 0.9  # 0.9 for format fuzzing and 0.5 or letter for data fuzzing. Now format fuzzing

    list_of_objects = poc.get_list_of_object(text)
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

        print('--- Generating text with seed:\n "' + obj_prefix + '"')
        sys.stdout.write(generated)

        if generated.endswith('endobj'):
            generated_obj_counter += 1

        if generated_obj_counter > generated_obj_max_number:
            stop_condition = True

        while not stop_condition:
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(obj_prefix):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index, prob, preds2 = sample(preds, diversity)
            p_fuzz = random.random()
            if p_fuzz > t_fuzz and preds2[next_index] > p_t:
                next_index = np.argmin(preds2)
                print('FUZZ DONE!')

            next_char = indices_char[next_index]

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

        # save generated text to file inside program
        dt = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S_')
        dir_name = './generated_results/pdfobjs_new/'
        file_name = 'gen_objs' + dt + 'epochs' + repr(epochs) + '_div' \
                    + repr(diversity) + '_step' + repr(step) + '.txt'
        poc.save_to_file(dir_name + file_name, generated)
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


def main(argv):
    """ the main function """
    train()


if __name__ == "__main__":
    main(sys.argv)
