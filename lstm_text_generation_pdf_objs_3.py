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
from incremental_update.config import learning_config


class FileFormatFuzzer(object):
    """ Main class for learn and fuzz process"""
    def __init__(self, maxlen=85, step=1, batch_size=128):
        # learning hyper-parameter
        self.maxlen = maxlen
        self.step = step
        self.batch_size = batch_size

        self.text_all = ''
        self.text_training = ''
        self.text_validation = ''
        self.text_test = ''

        self.chars = None
        self.char_indices = None
        self.indices_char = None

        # self.model = None

        self.load_dataset()

    def define_model(self, input_dim, output_dim):
        """build the model: a single LSTM layer # we need to deep it"""
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

    def load_dataset(self):
        """ load all 3 part of each dataset and building dictionary index """
        if learning_config['dataset_size'] == 'small':
            self.text_training = preprocess.load_from_file(learning_config['small_training_set_path'])
            self.text_validation = preprocess.load_from_file(learning_config['small_validation_set_path'])
            self.text_test = preprocess.load_from_file(learning_config['small_testing_set_path'])
        elif learning_config['dataset_size'] == 'medium':
            self.text_training = preprocess.load_from_file(learning_config['medium_training_set_path'])
            self.text_validation = preprocess.load_from_file(learning_config['medium_validation_set_path'])
            self.text_test = preprocess.load_from_file(learning_config['medium_testing_set_path'])
        elif learning_config['dataset_size'] == 'large':
            self.text_training = preprocess.load_from_file(learning_config['large_training_set_path'])
            self.text_validation = preprocess.load_from_file(learning_config['large_validation_set_path'])
            self.text_test = preprocess.load_from_file(learning_config['large_testing_set_path'])
        self.text_all = self.text_training + self.text_validation + self.text_test
        print('Total corpus length:', len(self.text_all))
        self.chars = sorted(list(set(self.text_all)))
        print('Total corpus chars:', len(self.chars))
        # print(chars)

        # Building dictionary index
        print('Building dictionary index ...')
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        # print(char_indices)
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        # print(indices_char)

    def generate_samples(self, text):
        """Cut the text in semi-redundant sequences of maxlen characters"""
        sentences = []  # list of all sentence as input
        next_chars = []  # list of all next chars as labels
        for i in range(0, len(text) - self.maxlen, self.step):  # arg2 why this?
            sentences.append(text[i: i + self.maxlen])
            # print(sentences)
            next_chars.append(text[i + self.maxlen])
            # print(next_chars)
        print('Number of semi sequences:', len(sentences))
        return sentences, next_chars

    def data_generator(self, sentences, next_chars):
        j = 0
        # print('Vectorization...')
        x = np.zeros((self.batch_size, self.maxlen, len(self.chars)), dtype=np.bool)
        y = np.zeros((self.batch_size, len(self.chars)), dtype=np.bool)
        while True:
            j = random.randint(0, len(sentences) - 1)
            for i, one_sample in enumerate(sentences[j: j + self.batch_size]):
                for t, char in enumerate(one_sample):
                    x[i, t, self.char_indices[char]] = 1
                y[i, self.char_indices[next_chars[i]]] = 1
            yield x, y
            # j += batch_size

    def train(self, epochs=1):
        """ Create and train deep model"""
        print('Generate training samples ...')
        sentences_training, next_chars_training = self.generate_samples(self.text_training)
        print('Generate validations samples ...')
        sentences_validation, next_chars_validation = self.generate_samples(self.text_validation)

        print('Build and compile model ...')
        model = self.define_model((self.maxlen, len(self.chars)), len(self.chars))
        optimizer = RMSprop(lr=0.02)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        print('Build training and validation data generators ...')
        ts_data_generator = self.data_generator(sentences_training, next_chars_training)
        vs_data_generator = self.data_generator(sentences_validation, next_chars_validation)

        print('Set #4 callback ...')
        dt = datetime.datetime.now().strftime('date_%Y%m%d_%H%M%S_')
        model_checkpoint_filepath = './model_checkpoint/'+dt+'epoch_{epoch:02d}_{val_loss:.4f}.h5'
        model_chekpoint = ModelCheckpoint(model_checkpoint_filepath, verbose=1)

        model_early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=1, verbose=0, mode='auto')

        model_tensorboard = TensorBoard(log_dir='./logs_tensorboard', histogram_freq=0, batch_size=self.batch_size,
                                        write_graph=True,write_grads=False, write_images=True, embeddings_freq=0,
                                        embeddings_layer_names=None, embeddings_metadata=None)
        model_csv_logger_filepath = './logs_csv/training_with_epoch_02.csv'
        model_csv_logger = CSVLogger(model_csv_logger_filepath, separator=',', append=False)

        def on_epoch_end(epoch, logs):
            nonlocal model
            nonlocal epochs
            self.generate_and_fuzz_new_samples(self, model, epoch, self.step)
        generate_and_fuzz_new_samples_callback = LambdaCallback(on_epoch_begin=None,
                                                                on_epoch_end=on_epoch_end,
                                                                on_batch_begin=None, on_batch_end=None,
                                                                on_train_begin=None, on_train_end=None)

        print('Start training ...')
        model.fit_generator(generator=ts_data_generator,
                            steps_per_epoch=len(sentences_training) // self.batch_size,
                            validation_data=vs_data_generator,
                            validation_steps=len(sentences_validation) // self.batch_size,
                            epochs=epochs,
                            callbacks=[model_chekpoint, model_early_stopping, model_tensorboard, model_csv_logger])

        # model.fit(x, y, batch_size=128, epochs= epoch_list[0], validation_split=0.2)  # why epochs=?
        # save(model,  epoch_list[0])
        # del model
        # model = load_model('./modelh5/lstm_text_generation_pdf_objs_1_20180214_235713_epochs10.h5')

    def generate_and_fuzz_new_samples(self, model, epoch, step):
        """ sampling the model and generate new object """
        # fuzzing hyperparameters
        # diversities = [i*0.10 for i in range(1,20,2)]
        diversities = [0.2, 0.5, 1.0, 1.2, 1.5, 1.8]
        generated_obj_max_number = 5
        generated_obj_max_allowed_len = 500
        t_fuzz = 0.9
        p_t = 0.9  # 0.9 for format fuzzing and 0.5 or letter for data fuzzing. Now format fuzzing
        # end of fuzzing parameters

        list_of_objects = preprocess.get_list_of_object(self.text_test)
        list_of_objects_with_maxlen = []
        for o in list_of_objects:
            if len(o) > self.maxlen:
                list_of_objects_with_maxlen.append(o)

        for diversity in diversities:
            obj_index = random.randint(0, len(list_of_objects_with_maxlen) - 1)
            generated_obj_counter = 0
            generated_obj_len_index = 0

            stop_condition = False
            print()
            print('-- Diversity:', diversity)

            # generated = ''
            obj_prefix = str(list_of_objects_with_maxlen[obj_index])[0: self.maxlen]  # len(sentence) equals 100 here

            generated = obj_prefix
            prob_vals = '100\n' * self.maxlen
            learnt_grammar = obj_prefix

            print('--- Generating ts_text with seed:\n "' + obj_prefix + '"')
            sys.stdout.write(generated)

            if generated.endswith('endobj'):
                generated_obj_counter += 1

            if generated_obj_counter > generated_obj_max_number:
                stop_condition = True

            while not stop_condition:
                x_pred = np.zeros((1, self.maxlen, len(self.chars)))
                for t, char in enumerate(obj_prefix):
                    x_pred[0, t, self.char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index, prob, preds2 = self.sample(preds, diversity)
                p_fuzz = random.random()
                if p_fuzz > t_fuzz and preds2[next_index] > p_t:
                    next_index = np.argmin(preds2)
                    print('FUZZ DONE!')

                next_char = self.indices_char[next_index]

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
            file_name = 'gen_objs' + dt + 'epochs' + repr(epoch) + '_div' \
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

    def no_sample(self):
        pass

    def sample_space(self):
        pass


def save_model_plot(model, epochs):
    """ save the model architecture plot """
    dt = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S_')
    # plot the model
    plot_model(model, to_file='./modelpic/date_' + dt + 'epochs_' + str(epochs) + '.png',
               show_shapes=True, show_layer_names=True)


def main(argv):
    """ the main function """
    fff = FileFormatFuzzer()
    fff.train(2)


if __name__ == "__main__":
    main(sys.argv)
