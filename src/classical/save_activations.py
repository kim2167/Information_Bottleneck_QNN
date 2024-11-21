import os
import keras
import random
import pickle
import numpy as np
import loggingreporter 
import matplotlib.pyplot as plt
from collections import namedtuple

def do_report(epoch):
    # Only log activity for some epochs.  Mainly this is to make things run faster.
    if epoch < 50:       # Log for all first 20 epochs
        return True
    elif epoch < 100:    # Then for every 5th epoch
        return (epoch % 4 == 0)
    elif epoch < 2000:    # Then every 10th
        return (epoch % 20 == 0)
    else:                # Then every 100th
        return (epoch % 100 == 0)

def generate_dataset(size_data):
    n = size_data
    basis_vector = np.zeros((n,n+1))
    for i in range(n):
        basis_vector[i,n-1-i] = 1
        if i %2 == 0:
            basis_vector[i,-1] = 0
        else:
            basis_vector[i,-1] = 1

    np.savetxt('./orthog' + str(size_data) + '.txt', basis_vector, fmt="%d")

def shuffle_data(X,Y):
    random.seed(12)
    list_suffle = []
    for i in range(0, len(X)):
        list_suffle.append([X[i],Y[i]])

    random.shuffle(list_suffle)
    
    X_shuffled = []
    Y_shuffled = []

    for i in range(0, len(X)):
        X_shuffled.append(list_suffle[i][0])
        Y_shuffled.append(list_suffle[i][1])
    
    return np.array(X_shuffled), np.array(Y_shuffled)

def get_dataset(size_data, ratio_train, gen_dataset=False):

    if(gen_dataset):
        generate_dataset(size_data)

    data = np.loadtxt('./orthog' + str(size_data) + '.txt')
    X = np.array(data[:, :-1])
    Y = np.array(data[:, -1])
    
    nb_classes = 2

    num_train = int(ratio_train * size_data)
    print("num_train", num_train)

    X_train = X[:num_train]
    X_test = X[num_train:]
    y_train = Y[:num_train]
    y_test = Y[num_train:]


    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    X_test  = np.reshape(X_test , [X_test.shape[0] , -1])

    Y_train = keras.utils.np_utils.to_categorical(y_train, nb_classes).astype('float32')
    Y_test  = keras.utils.np_utils.to_categorical(y_test, nb_classes).astype('float32')
    # print(Y_train[0])

    Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])
    trn = Dataset(X_train, Y_train, y_train, nb_classes)
    tst = Dataset(X_test , Y_test, y_test, nb_classes)

    del X_train, X_test, Y_train, Y_test, y_train, y_test
    return trn, tst

def save_activations(input_set):

    activation_func, learning_rate, layer_dim, batch_size, num_epochs, ratio_train, base_path, size_data, gen_dataset = input_set

    cfg = {}
    cfg['SGD_BATCHSIZE']    = batch_size
    cfg['NUM_EPOCHS']       = num_epochs
    cfg['FULL_MI']          = True

    cfg['ACTIVATION'] = activation_func
    cfg['SGD_LEARNINGRATE'] = learning_rate
    cfg['LAYER_DIMS'] = layer_dim

    ARCH_NAME =  '-'.join(map(str,cfg['LAYER_DIMS']))
    trn, tst = get_dataset(size_data, ratio_train, gen_dataset)

    # Where to save activation and weights data
    cfg['SAVE_DIR'] = base_path + str(ratio_train) +'/data_epoch/' + str(size_data) + '/' + ARCH_NAME + '/' + cfg['ACTIVATION'] + '_' + ARCH_NAME + '_' + str(cfg['SGD_LEARNINGRATE'])


    input_layer = keras.layers.Input(shape=(trn.X.shape[1],))
    clayer = input_layer
    for n in cfg['LAYER_DIMS']:
        clayer = keras.layers.Dense(n, 
                                    activation=cfg['ACTIVATION'],
                                    kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=1/np.sqrt(float(n)), seed=None),
                                    bias_initializer='zeros'
                                )(clayer)
    output_layer = keras.layers.Dense(trn.nb_classes, activation='softmax')(clayer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    optimizer = keras.optimizers.Adam(learning_rate=cfg['SGD_LEARNINGRATE'])

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    reporter = loggingreporter.LoggingReporter(cfg=cfg, 
                                        trn=trn, 
                                        tst=tst, 
                                        do_save_func=do_report)
    r = model.fit(x=trn.X, y=trn.Y, 
                verbose    = 2, 
                batch_size = cfg['SGD_BATCHSIZE'],
                epochs     = cfg['NUM_EPOCHS'],
                validation_data=(tst.X, tst.Y),
                callbacks  = [reporter,])
    
    accuracy_train = r.history['accuracy']
    accuracy_test = r.history['val_accuracy']

    acc_dict = {"accuracy_train":accuracy_train, "accuracy_test":accuracy_test}
    acc_file_name = base_path + str(ratio_train) +'/data_acc/' + str(size_data) + '/' + ARCH_NAME + '/' + cfg['ACTIVATION'] + '_' + ARCH_NAME + '_' + str(cfg['SGD_LEARNINGRATE'])
    if not os.path.exists(base_path + str(ratio_train) +'/data_acc/' + str(size_data) + '/' + ARCH_NAME + '/'):
        # print("Making directory", self.cfg['SAVE_DIR'])
        os.makedirs(base_path + str(ratio_train) +'/data_acc/' + str(size_data) + '/' + ARCH_NAME + '/')

    with open(acc_file_name + ".pickle", 'wb') as handle:
        pickle.dump(acc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.plot(acc_dict["accuracy_train"][:500],color="orange", label="train")
    plt.plot(acc_dict["accuracy_test"][:500], color="blue", label="test")
    plt.legend()
    plt.savefig(acc_file_name+".png", bbox_inches='tight')

    plt.clf()
    plt.close()