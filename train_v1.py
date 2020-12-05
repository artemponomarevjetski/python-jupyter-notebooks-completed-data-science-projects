"""
artemp: to prepare the code for execution on a laptop do this
% cd Desktop/ML_task/
% python3 -m venv  .
% source ./venv activate
% pip3 install --upgrade pip
% pip3 install -r requirements.txt
# if the above does not work, install the libraries individually
% pip3 install numpy
% pip3 install pandas 
% python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
% pip3 install opencv-python
# use command 'python3 train_v2.py -f data', or with more flags; the new flag, -a, is to choose between 'SGD' and 'Adam' optimizer
% source ./venv deactivate
"""
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import argparse

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras import optimizers


def load_data_folder(folder, set_index=-1):
    """
    artemp: This function prepares a dataframe for an individual folder, set?, 
    with filenames in that folder and other information
    """

    csv_file_name = folder / 'sequence_info_by_frame.csv'
    df = pd.read_csv(csv_file_name)
    img_file_names = sorted(folder.glob('seq???.png'))
    df['set_index'] = set_index
    df['file_name'] = [str(f.resolve()) for f in img_file_names]

    return df


def load_all_data(base_folder, split_by='set', seed=None):
    """
    artemp: this function loads pixel values from images using cv2.imread() function; 
    cv2.IMREAD_GRAYSCALE argument is present because the images are in grey scale;
    it randomizes the order of folders with images and assigns the folders to the train, 
    test and validate data sets;
    if 'seed' is specified, the execution will start with a given seed, this allows to repeat
    the same calculation; two seeds need to be fixed, one in numpy library, the other in
    tensorflow library. I also fixed the choice of the train/val/test sets to better understand 
    model outputs.
    """
    set_names = sorted(base_folder.glob('set?'))
    columns = ['frame', 'object0_present', 'object1_present',
               'set_index', 'file_name']
    df = pd.DataFrame(columns=columns)

    print()
    print('loading all data into dataframe ...')
    print()

    for idx, folder in enumerate(set_names):

        print('loading data from set {} of {}'.format(idx+1, len(set_names)))
        df_partial = load_data_folder(folder, set_index=idx)
        df = pd.concat([df, df_partial])

    df['is_train'] = False
    df['is_val'] = False
    df['is_test'] = False
# artemp: df dataframe contains indexes to specify train/val/test data sets
    print('df = ', df)

    print('seed = ', seed)
# artemp: this code, as provided, can't reproduce the result by overriding the default value
# for 'seed' with a certain value, as there is randomness in tensorflow library
    if seed: 
        np.random.seed(seed)

# artemp: the only way to reproduce the same result is the code below
# comment the code below to enable randomness again
    from numpy.random import seed
    seed(1)
    from tensorflow import set_random_seed
    set_random_seed(2)
# artemp: this block of code fixes the seed both in the numpy and tensorflow libraries
    
    print()
    print('loading data into memory ...')

    all_data_imgs = np.array([cv2.imread(fname, cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
                              for fname in df.file_name])
    print('imgs shape: ', all_data_imgs.shape)
    all_data_labels = df.loc[:, ['object0_present', 'object1_present']].values
    print('labels shape: ', all_data_labels.shape)
    print('done.')
    print()

    print('split_by = ', split_by)
    print()
# artemp: it seems that in all examples 'split_by=='set''
    if split_by == 'set':

        set_indices = df['set_index'].unique()
        set_indices.sort()
        n_sets = len(set_indices)

        reordered_sets = np.random.choice(set_indices, n_sets, replace=False)
        test_indices = reordered_sets[-1:]
        val_indices = reordered_sets[-2:-1]
        train_indices = reordered_sets[:-2]
        """
        artemp: this kind of array slicing, -1, means: "one from the end";
        for example, arr = np.array([1, 2, 3, 4, 5, 6, 7])
        print(arr[-3:-1])
        results in [5, 6]
        """
        
# artemp: test-run the model with a fixed choice for the train, validation and test data sets
# comment next 3 line to enable random choice of sets for training/validation/testing.
        test_indices =  [4]
        val_indices = [3]
        train_indices = [2, 0, 1]
        print('folders chosen for test data: ', test_indices)
        print('folders chosen for validation data: ', val_indices)
        print('folders chosen for train data: ', train_indices)
        print()

        df['is_train'] = df['set_index'].isin(train_indices)
        df['is_val'] = df['set_index'].isin(val_indices)
        df['is_test'] = df['set_index'].isin(test_indices)

    elif split_by == 'random':

        n_rows = len(df)
        train_prob = 0.6
        val_prob = 0.2
        test_prob = 0.2
        data_group_index = np.random.choice(3, n_rows, p=[train_prob, val_prob, test_prob])

        df['is_train'] = (data_group_index == 0)
        df['is_val'] = (data_group_index == 1)
        df['is_test'] = (data_group_index == 2)

    else:

        print('split by {} not implemented'.format(split_by))

    x_train = all_data_imgs[df['is_train']]
    y_train = all_data_labels[df['is_train']].astype(int)

    x_val = all_data_imgs[df['is_val']]
    y_val = all_data_labels[df['is_val']].astype(int)

    x_test = all_data_imgs[df['is_test']]
    y_test = all_data_labels[df['is_test']].astype(int)

    train_data = (x_train, y_train)
    val_data = (x_val, y_val)
    test_data = (x_test, y_test)

    return train_data, val_data, test_data


def build_model(n_features_conv, input_shape, n_features_dense=[128, 64],
                n_output_classes=2):
    """
    artemp: this function builds a convolutional neural net
    """
    model = Sequential()

    block_index = 0
    conv_index = 0
    print(n_features_conv[block_index])
    conv = Conv2D(n_features_conv[block_index], 3, padding='same', strides=(1, 1),
                  name='block{}_conv{}'.format(block_index+1, conv_index+1),
                  activation='relu', input_shape=input_shape)
    model.add(conv)
    pool = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                     name='maxpool{}'.format(block_index))
    model.add(pool)

    block_index += 1
    conv_index = 1
    print(n_features_conv[block_index])
    conv = Conv2D(n_features_conv[block_index], 3, padding='same', strides=(1, 1),
                name='block{}_conv{}'.format(block_index+1, conv_index+1),
                activation='relu')
    model.add(conv)
    pool = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                    name='maxpool{}'.format(block_index))
    model.add(pool)

    block_index += 1
    conv_index = 1
    conv = Conv2D(n_features_conv[block_index], 3, padding='same', strides=(1, 1),
                name='block{}_conv{}'.format(block_index+1, conv_index+1),
                activation='relu')
    model.add(conv)
    pool = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                    name='maxpool{}'.format(block_index))
    model.add(pool)

    block_index += 1
    conv_index = 1
    conv = Conv2D(n_features_conv[block_index], 3, padding='same', strides=(1, 1),
                name='block{}_conv{}'.format(block_index+1, conv_index+1),
                activation='relu')
    model.add(conv)
    pool = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                    name='maxpool{}'.format(block_index))
    model.add(pool)

    model.add(Flatten(name='flatten'))

    dense_block_index = 0
    dense = Dense(n_features_dense[dense_block_index],
                  name='dense{}'.format(dense_block_index+1),
                  activation='relu')
    model.add(dense)

    dense_block_index += 1
    dense = Dense(n_features_dense[dense_block_index],
                  name='dense{}'.format(dense_block_index+1),
                  activation='relu')
    model.add(dense)
# artemp: dense layer means that every neuron is connected to all neurons in the next layer
    print('number of convolutonal layers', block_index+1)
    print('number of dense layers', dense_block_index+1)

    classifier = Dense(n_output_classes,
                       name='classifier',
                       activation='sigmoid')
    model.add(classifier)

    return model


if __name__ == "__main__":
    """
    artemp: I added new code for 'Adam' optimizer; execute it like this: % python3 train_v2.py -f data -o 'Adam'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', help='number of training epochs',
                        type=int, default=10)
    parser.add_argument('-b', '--batch_size', help='training batch size',
                        type=int, default=5)
    parser.add_argument('-f', '--folder', help='root data folder',
                        type=str, required=True)
    parser.add_argument('-l', '--learning_rate', help='learning rate',
                        type=float, default=1e-4)  
    parser.add_argument('-o', '--optimizer', help='optimizer',
                        type=str, default='SGD')

    args = parser.parse_args()
    n_epochs = args.epochs
    batch_size = args.batch_size
    lrate = args.learning_rate
    optimizer = args.optimizer

    base_folder = Path(args.folder)
    if not base_folder.exists():
        print('please include a valid path to data files')

    # load the data
    print()
    print('loading the data ...')
    train_data, val_data, test_data = load_all_data(base_folder)
    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data
    print()

    INPUT_SHAPE = (150, 128, 1)
    NUMBER_FEATURES_CONV = [16, 32, 64, 128]
    NUMBER_FEATURES_DENSE = [64, 32]

    NUMBER_FEATURES_CONV = [8, 16, 32, 64]
    NUMBER_FEATURES_DENSE = [64, 32]
    
    print('model has the following parameters: ')
    print('INPUT_SHAPE =  ', INPUT_SHAPE)
    print('NUMBER_FEATURES_CONV =', NUMBER_FEATURES_CONV)
    print('NUMBER_FEATURES_DENSE = ', NUMBER_FEATURES_DENSE)
    print()

    # load the model
    print('loading the model ...')
    model = build_model(NUMBER_FEATURES_CONV, input_shape=INPUT_SHAPE,
                        n_features_dense=NUMBER_FEATURES_DENSE)
    print()

    # set optimizer and loss
    opt = None
    if optimizer == 'SGD':
        opt = optimizers.SGD(lr=lrate, momentum=0.9, nesterov=True)
# artemp: In this case CNN is trained using the stochastic gradient descent optimization algorithm
    elif optimizer == 'Adam':
        opt = optimizers.Adam(lr=0.0001, decay=1e-6)
    else:
        print('Specify the optimizer from [''Adam'', ''SGD'']')

    print('compiling the model ...')
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    print()

    # model info
    print('Model summary: ')
    print(model.summary())
    print()

    # train
    print('training the model ...')
    training = model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size,
              validation_data=(x_val, y_val))
# artemp: I added history output to visualize training and validation loss
    import matplotlib.pyplot as plt
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.savefig('test.png')
    plt.show()

    # test
    print()
    print()
    print('now evaluating on the test set:')
    final_result = model.evaluate(x_test, y_test)

    print('final_result = ', final_result)
    print()
    print('loss = ', final_result[0])
    print()
    print('accuracy = ', final_result[1])
    print()
