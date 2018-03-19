import numpy as np
import pdf_object_concat as poc

class DataGenerator(object):
    """Generates data for Keras"""
    def __init__(self, dim_x = 32, dim_y = 32, dim_z = 32, batch_size = 32, shuffle = True):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle

        """ for using with data generator"""
        trainset_path = './trainset/pdfobjs.txt'
        trainset_path = './trainset/pdf_object_trainset_100_to_500_percent10.txt'
        self.text = poc.load_from_file(trainset_path)
        print('corpus length:', len(self.text))

        self.chars = sorted(list(set(self.text)))
        print('Total chars:', len(self.chars))

        # Vectorization
        print('Building dictionary index ...')
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        # print(char_indices)
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        # print(indices_char)

        # cut the text in semi-redundant sequences of maxlen characters
        self.maxlen = 100  # Good idea: use ave_object_len to determine this hyper-parameter

    def generate(self, labels, list_IDs):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                # Generate data
                X, y = self.__data_generation(labels, list_IDs_temp)

                yield X, y

    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, labels, list_IDs_temp):
        'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z, 1))
        y = np.empty((self.batch_size), dtype = int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            text = poc.load_from_file('./npysamples/IDs/' + ID)
            labe = poc.load_from_file('./npysamples/Labels/' + ID)
            x = np.zeros((1, self.maxlen, len(self.chars)), dtype=np.bool)  # input x
            y = np.zeros((1, len(self.chars)), dtype=np.bool)  # output label y
            for t, char in enumerate(text):
                x[0, t, self.char_indices[char]] = 1
            y[0, self.char_indices[next_chars[i]]] = 1

            X[i, :, :, :, 0] = np.load(ID + '.npy')

            # Store class
            y[i] = labels[ID]

        return X, sparsify(y)


def sparsify(y):
    'Returns labels in binary NumPy array'
    n_classes = 96 # Enter number of classes
    return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                     for i in range(y.shape[0])])