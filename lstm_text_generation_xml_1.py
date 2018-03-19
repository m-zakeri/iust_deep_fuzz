'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io

import datetime
from keras.utils import plot_model

# read training data from file (data set)
#path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
#text = io.open(path, encoding='utf-8').read().lower()
text = open('trainset/larg.xml', encoding="utf8").read()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
print(chars)

char_indices = dict((c, i) for i, c in enumerate(chars))
print(char_indices)
indices_char = dict((i, c) for i, c in enumerate(chars))
print(indices_char)


# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = [] # list of all sentence
next_chars = [] # list of all next chars
for i in range(0, len(text) - maxlen, step): # arg2 why this?
    sentences.append(text[i: i + maxlen])
    #print(sentences)
    next_chars.append(text[i + maxlen])
    #print(next_chars)
    

print('semi sequences:', len(sentences))

print('One-Hot vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool) #input x
y = np.zeros((len(sentences), len(chars)), dtype=np.bool) #output label y
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM layer # we need to deep it
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#sys.exit()

# plot the model
dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
plot_model(model, to_file = 'lstm_text_generation_edited1_' + dt + '.png', show_shapes=True, show_layer_names=True)

# lower temperature will cause the model to make more likely,
# but also more boring and conservative predictions.
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 50):
    print()
    print('-' * 50)
    print('- Iteration', iteration)
    model.fit(x, y, batch_size=128, epochs=3) # why epochs=?

     #choose random start index for seed sentence from trian set
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2, 1.5]:
        print()
        print('-- Diversity:', diversity)

        #generated = ''
        sentence = text[start_index: start_index + maxlen] # len(sentence) equals 40 here
        generated = sentence
        print('-- Generating text with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(500):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        #save generated text to file inside program
        file_name = 'XMLs/' + 'step' + repr(step) + '_itr' + repr(iteration) + '_div' + repr(diversity) + '.txt'
        with open(file_name, 'w') as generated_text_file:
             generated_text_file.write(generated)
        print()
        

# Save model (requires HDF5 and h5py)
dt = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model.save('lstm_text_generation_edited1_'+ dt + '.h5')


