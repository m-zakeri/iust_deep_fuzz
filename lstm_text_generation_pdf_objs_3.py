"""
PDF OBJ 3
-New in this version:
-- Add support for training in large dataset with the help of python generators.
-- Add callbacks to log most of training time events.
-- File and directory now mange by code in appropriate manner for each train run.
-- Add class FileFormatFuzz to do learn and fuzz process in one script.
--
-Note: The ability of training small dataset in memory with model.fit() method was include in this version.

"""

from __future__ import print_function

__version__ = '0.3'
__author__ = 'Morteza'

import sys
import os
import datetime
import random
import numpy as np

from keras.models import load_model
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, LambdaCallback
from keras.utils import plot_model

import pdf_object_preprocess as preprocess
from config import learning_config
import deep_models


class FileFormatFuzzer(object):
    """
    Main class for learn and fuzz process
    """
    def __init__(self, maxlen=85, step=1, batch_size=128):
        """

        :param maxlen:
        :param step:
        :param batch_size:
        """
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
        model, model_name = deep_models.model_3(input_dim, output_dim)
        return model, model_name

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
        print('Number of semi sequences or samples:', len(sentences))
        return sentences, next_chars

    def data_generator(self, sentences, next_chars):
        """Batch data generator for large dataset not fit completely in memory"""
        j = 0
        # print('Vectorization...')
        x = np.zeros((self.batch_size, self.maxlen, len(self.chars)), dtype=np.bool)
        y = np.zeros((self.batch_size, len(self.chars)), dtype=np.bool)
        while True:
            j = random.randint(0, len(sentences) - (self.batch_size+1))
            for i, one_sample in enumerate(sentences[j: j + self.batch_size]):
                for t, char in enumerate(one_sample):
                    x[i, t, self.char_indices[char]] = 1
                y[i, self.char_indices[next_chars[i]]] = 1
            yield x, y
            # j += batch_size

    def data_generator_in_memory(self, sentences, next_chars):
        """All data generate for small dataset fit completely in memory"""
        x = np.zeros((len(sentences), self.maxlen, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)
        for i, one_sample in enumerate(sentences):
            for t, char in enumerate(one_sample):
                x[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1
        return x, y

    def train(self, epochs=1):
        """ Create and train deep model"""
        # Start time of training
        dt = datetime.datetime.now().strftime('_date_%Y-%m-%d_%H-%M-%S_')

        print('Generate training samples ...')
        sentences_training, next_chars_training = self.generate_samples(self.text_training)
        print('Generate validations samples ...')
        sentences_validation, next_chars_validation = self.generate_samples(self.text_validation)

        print('Build and compile model ...')
        model, model_name = self.define_model((self.maxlen, len(self.chars)), len(self.chars))
        optimizer = RMSprop(lr=0.01)  # [0.01, 0.02, 0.05, 0.1]
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        print(model_name, ' summary ...')
        model.summary()

        print(model_name, ' count_params ...')
        print(model.count_params())
        # input()

        print('Set #5 callback ...')
        # callback #1 EarlyStopping
        model_early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, verbose=1, mode='auto')

        # callback #2 ModelCheckpoint
        # Create a directory for each training process to keep model checkpoint in .h5 format
        dir_name = './model_checkpoint/pdfs/' + model_name + dt + 'epochs_' + str(epochs) + '/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = dir_name + model_name + dt + 'epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'
        model_chekpoint = ModelCheckpoint(file_name, verbose=1)

        # callback #3 TensorBoard
        dir_name = './logs_tensorboard/pdfs/' + model_name + dt + 'epochs_' + str(epochs) + '/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        model_tensorboard = TensorBoard(log_dir=dir_name, histogram_freq=0, batch_size=self.batch_size,
                                        write_graph=True, write_grads=False, write_images=True, embeddings_freq=0,
                                        embeddings_layer_names=None, embeddings_metadata=None)

        # callback #4 CSVLogger
        # Create a directory and an empty csv file within to save mode csv log.
        dir_name = './logs_csv/pdfs/' + model_name + dt + 'epochs_' + str(epochs) + '/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = dir_name + model_name + dt + '_epochs_' + str(epochs) + '_step_' + str(self.step) + '.csv'
        open(file_name, mode='a', newline='').close()
        model_csv_logger = CSVLogger(file_name, separator=',', append=True)

        # callback #5 LambdaCallback
        dir_name = './generated_results/pdfs/' + model_name + dt + 'epochs_' + str(epochs) + '/'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        def on_epoch_end(epoch, logs):
            nonlocal model
            nonlocal epochs
            nonlocal model_name
            nonlocal dir_name
            print('Sampling model and save results ... ')
            self.generate_and_fuzz_new_samples(model=model, model_name=model_name,
                                               epochs=epochs, step=self.step, maxlen=self.maxlen,
                                               len_chars=len(self.chars), dir_name=dir_name)
        generate_and_fuzz_new_samples_callback = LambdaCallback(on_epoch_begin=None,
                                                                on_epoch_end=on_epoch_end,
                                                                on_batch_begin=None, on_batch_end=None,
                                                                on_train_begin=None, on_train_end=None)

        if learning_config['dataset_size'] == 'very_small':
            print('Start training on small dataset ...')
            x, y = self.data_generator_in_memory(sentences_training, next_chars_training)
            model.fit(x, y, batch_size=self.batch_size, epochs=epochs, validation_split=0.2,
                      callbacks=[model_chekpoint, model_early_stopping, model_tensorboard, model_csv_logger,
                                 generate_and_fuzz_new_samples_callback])
        else:
            print('Build training and validation data generators ...')
            ts_data_generator = self.data_generator(sentences_training, next_chars_training)
            vs_data_generator = self.data_generator(sentences_validation, next_chars_validation)

            print('Start training on large dataset ...')
            model.fit_generator(generator=ts_data_generator,
                                # steps_per_epoch=200,
                                steps_per_epoch=len(sentences_training) // self.batch_size,  # 1000,
                                validation_data=vs_data_generator,
                                validation_steps=len(sentences_validation) // self.batch_size,  # 100,
                                # validation_steps=10,
                                epochs=epochs,
                                callbacks=[model_chekpoint, model_early_stopping, model_tensorboard, model_csv_logger,
                                           generate_and_fuzz_new_samples_callback])
    # end of train method

    def generate_and_fuzz_new_samples(self,
                                      model=None,
                                      model_name='model_1',
                                      epochs=1, step=1, maxlen=100, len_chars=96,
                                      dir_name=None):
        """ sampling the model and generate new object """
        # fuzzing hyperparameters
        # diversities = [i*0.10 for i in range(1,20,2)]
        diversities = [0.2, 0.5, 1.0, 1.2, 1.5, 1.8]
        diversities = [1]
        generated_obj_total = 5  # [100, 1000]
        generated_obj_with_same_prefix = 5  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        generated_obj_max_allowed_len = random.randint(400, 500)
        parasite_chars_set = {'s', 't', 'r', 'e', 'a', 'm'}   # set(['s', 't', 'r', 'e', 'a', 'm'])
        t_fuzz = 0.9
        p_t = 0.9  # 0.9 for format fuzzing and 0.5 or little than 0.5 for data fuzzing. Now format fuzzing
        # end of fuzzing hyperparameters

        testset_objects_list = preprocess.get_list_of_object(self.text_test)
        testset_object_gt_maxlen_list = []
        for obj in testset_objects_list:
            if len(obj) > maxlen:
                testset_object_gt_maxlen_list.append(obj)
        print('len filtered testset:', len(testset_object_gt_maxlen_list))
        for diversity in diversities:
            generated_total = ''
            for q in range(round(generated_obj_total/generated_obj_with_same_prefix)):
                obj_index = random.randint(0, len(testset_object_gt_maxlen_list) - 1)
                generated_obj_counter = 0
                generated_obj_len_index = 0
                stop_condition = False
                print()
                print('-- Diversity:', diversity)

                obj_prefix = str(testset_object_gt_maxlen_list[obj_index])[0: maxlen]
                generated = obj_prefix
                prob_vals = '1 ' * self.maxlen
                learnt_grammar = obj_prefix

                print('--- Generating ts_text with seed:\n "' + obj_prefix + '"')
                sys.stdout.write(generated)

                if generated.endswith('endobj'):
                    generated_obj_counter += 1

                if generated_obj_counter > generated_obj_with_same_prefix:
                    stop_condition = True

                while not stop_condition:
                    x_pred = np.zeros((1, maxlen, len_chars))
                    for t, char in enumerate(obj_prefix):
                        x_pred[0, t, self.char_indices[char]] = 1.

                    preds = model.predict(x_pred, verbose=0)[0]
                    next_index, prob, preds2 = self.sample(preds, diversity)
                    next_char = self.indices_char[next_index]
                    if next_char not in parasite_chars_set:
                        p_fuzz = random.random()
                        if p_fuzz > t_fuzz and preds2[next_index] > p_t:
                            next_index = np.argmin(preds2)
                            print('Character fuzzed!')
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

                    if generated_obj_counter > generated_obj_with_same_prefix:
                        stop_condition = True

                    sys.stdout.write(next_char)
                    sys.stdout.flush()
                generated_total += generated
            # save generated_result to file inside program

            file_name = 'diversity_' + repr(diversity) + '_epochs_' + repr(epochs) + '_step_' + repr(step) + '.txt'
            preprocess.save_to_file(dir_name + file_name, generated_total)
            # preprocess.save_to_file(dir_name + file_name + 'probabilities.txt', prob_vals)
            # preprocess.save_to_file(dir_name + file_name + 'learntgrammar.txt',learnt_grammar)

    # lower temperature will cause the model to make more likely,
    # but also more boring and conservative predictions.
    def sample(self, preds, temperature=1.0):
        """helper function to sample an index from a probability array"""
        # print('raw predictions = ', preds)
        preds = np.asarray(preds).astype('float64')

        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        """sampling with numpy functions:"""
        probas = np.random.multinomial(1, preds, 1)
        # print()
        # print('sanitize predictions = ', preds)
        return np.argmax(probas), probas, preds

    def no_sample(self):
        pass

    def sample_space(self):
        pass

    def save_model_plot(self, model, epochs):
        """ save the model architecture plot """
        dt = datetime.datetime.now().strftime('_%Y%m%d_%H%M%S_')
        # plot the model
        plot_model(model, to_file='./modelpic/date_' + dt + 'epochs_' + str(epochs) + '.png',
                   show_shapes=True, show_layer_names=True)


def main(argv):
    """ the main function """
    fff = FileFormatFuzzer(maxlen=85, step=1, batch_size=256)
    fff.train(epochs=10)
    previous_model_dir = './model_checkpoint/best_models/'
    previous_model_name = 'date_20180325_200701_epoch_02_7.3107.h5'
    previous_model_path = previous_model_dir + previous_model_name
    model = load_model(previous_model_path)
    # fff.generate_and_fuzz_new_samples(model=model, model_name='best_models', maxlen=85, len_chars=96)


if __name__ == "__main__":
    main(sys.argv)
