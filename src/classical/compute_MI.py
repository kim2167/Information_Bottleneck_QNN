
import os
import kde
import utils
import keras
import random
import numpy as np
import keras.backend as K
from six.moves import cPickle
import matplotlib.pyplot as plt
from collections import namedtuple
from collections import defaultdict, OrderedDict

def plot_type_1(measures, PLOT_LAYERS, COLORBAR_MAX_EPOCHS, infoplane_measure, name_output):

    sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
    sm._A = []

    fig=plt.figure(figsize=(15,5))
    cnt = 1
    for actndx, (activation, vals) in enumerate(measures.items()):
        epochs = sorted(vals.keys())
        if not len(epochs):
            continue

        plt.subplot(1,3,cnt)
        cnt += 1
        for epoch in epochs:
            c = sm.to_rgba(epoch)
            xmvals = np.array(vals[epoch]['MI_XM_'+infoplane_measure])[PLOT_LAYERS]
            ymvals = np.array(vals[epoch]['MI_YM_'+infoplane_measure])[PLOT_LAYERS]

            plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
            plt.scatter(xmvals, ymvals, s=20, facecolors=[c for _ in PLOT_LAYERS], edgecolor='none', zorder=2)

        plt.xlabel('I(X;M)')
        plt.ylabel('I(Y;M)')
        plt.title(activation)
        
    cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8]) 
    plt.colorbar(sm, label='Epoch', cax=cbaxes)
    plt.tight_layout()
    plt.savefig(name_output, bbox_inches='tight')

    plt.clf()
    plt.close()

def plot_type_2(Sx, Sy, name_output):
# def plot_type_2(measures, PLOT_LAYERS, COLORBAR_MAX_EPOCHS, infoplane_measure, name_output):
    
    fig = plt.figure(figsize=(16,12))

    list_color = ["red", "orange", "blue", "green", "purple", "black", "brown", "pink"]

    for i in range(0, 6):
        i_str = str(i+1)

        plt.scatter(Sx[i], Sy[i],color=list_color[i], label="layer"+i_str)
        # plt.plot(Sx[i], Sy[i],color=list_color[i], label="layer"+i_str,linestyle='dashed')

    # # plt.legend()
        plt.legend(fontsize=15)
    plt.xlabel('I(X;T)', fontsize = 20) 
    plt.ylabel('I(Y;T)', fontsize = 20) 
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.savefig(name_output, bbox_inches='tight')
    plt.clf()
    plt.close()

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

    nb_classes = 2
    data = np.loadtxt('./orthog' + str(size_data) + '.txt')
    X = np.array(data[:, :-1])
    Y = np.array(data[:, -1])

    num_train = int(ratio_train * size_data)
    X_train = X[:num_train]
    X_test = X[num_train:]
    y_train = Y[:num_train]
    y_test = Y[num_train:]

    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    X_test  = np.reshape(X_test , [X_test.shape[0] , -1])

    Y_train = keras.utils.np_utils.to_categorical(y_train, nb_classes).astype('float32')
    Y_test  = keras.utils.np_utils.to_categorical(y_test, nb_classes).astype('float32')
    # print(Y_train[0])

    print("y_test", len(y_test), len(X_train))
    Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])
    trn = Dataset(X_train, Y_train, y_train, nb_classes)
    tst = Dataset(X_test , Y_test, y_test, nb_classes)

    # del X_train, X_test, Y_train, Y_test, y_train, y_test
    return trn, tst

def compute_MI(input_set):
    # load data network was trained on

    learning_rate, g_noise, arch_layer, size_data, ratio_train, gen_dataset, base_path, do_type_1, do_type_2 = input_set
    trn, tst = get_dataset(size_data, ratio_train, gen_dataset)

    # calc MI for train and test. Save_activations must have been run with cfg['FULL_MI'] = True
    # FULL_MI = True
    FULL_MI = False

    # Which measure to plot
    infoplane_measure = 'upper'

    DO_LOWER = (infoplane_measure == 'lower')   # Whether to compute lower bounds also
    
    NUM_LABELS = 2
    MAX_EPOCHS = 1000
    COLORBAR_MAX_EPOCHS = 1000

    # Directories from which to load saved layer activity
    ARCH = arch_layer
    DIR_TEMPLATE = '%%s_%s'%ARCH
    noise_variance = g_noise                  # Added Gaussian noise variance

    lr = learning_rate
    path_output = base_path + str(ratio_train) + '/plots/'
    if not os.path.exists(path_output):
        os.mkdir(path_output)
    path_output += str(size_data) + '/'
    if not os.path.exists(path_output):
        os.mkdir(path_output)
    path_output += ARCH 
    if not os.path.exists(path_output):
        os.mkdir(path_output)
    path_output += '/lr_' + str(lr) + '/'
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    name_output_1 = path_output + ('infoplane_'+ARCH) + '_' + str(lr) + '_' + str(noise_variance) + "_type_1.png"
    name_output_2 = path_output + ('infoplane_'+ARCH) + '_' + str(lr) + '_' + str(noise_variance) + "_type_2.png"
    if(os.path.isfile(name_output_1)):
        print(('infoplane_'+ARCH) + '_' + str(lr) + '_' + str(noise_variance) + "_type_1.png", "already exists!")
        do_type_1 = False

    if(os.path.isfile(name_output_2)):
        print(('infoplane_'+ARCH) + '_' + str(lr) + '_' + str(noise_variance) + "_type_2.png", "already exists!")
        do_type_2 = False

    if(do_type_1 or do_type_2):
        print("Working on", ('infoplane_'+ARCH) + '_' + str(lr) + '_' + str(noise_variance) )
        # Functions to return upper and lower bounds on entropy of layer activity

        Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder 
        entropy_func_upper = K.function([Klayer_activity,], [kde.entropy_estimator_kl(Klayer_activity, noise_variance),])
        entropy_func_lower = K.function([Klayer_activity,], [kde.entropy_estimator_bd(Klayer_activity, noise_variance),])

        print(entropy_func_upper)
        # nats to bits conversion factor
        nats2bits = 1.0/np.log(2) 

        # Save indexes of tests data for each of the output classes
        saved_labelixs = {}

        y = tst.y
        Y = tst.Y
        if FULL_MI:
            full = utils.construct_full_dataset(trn,tst)
            y = full.y
            Y = full.Y

        print("Y", len(Y), len(y))
        for i in range(NUM_LABELS):
            saved_labelixs[i] = y == i


        labelprobs = np.mean(Y, axis=0)

        PLOT_LAYERS    = None     # Which layers to plot.  If None, all saved layers are plotted 

        # Data structure used to store results
        measures = OrderedDict()
        measures['tanh'] = {}
        measures['sigmoid'] = {}
        # measures['linear'] = {}
        # measures['softplus'] = {}

        Sx = []
        Sy = []
        for activation in measures.keys():
            cur_dir = base_path + str(ratio_train) +'/data_epoch/' + str(size_data) + '/' + ARCH  + '/' + DIR_TEMPLATE % activation + '_' + str(lr)
            if not os.path.exists(cur_dir):
                print("Directory %s not found" % cur_dir)
                continue
                
            # Load files saved during each epoch, and compute MI measures of the activity in that epoch
            print('*** Doing %s ***' % cur_dir)
            for epochfile in sorted(os.listdir(cur_dir)):
                if not epochfile.startswith('epoch'):
                    continue
                    
                fname = cur_dir + "/" + epochfile
                with open(fname, 'rb') as f:
                    d = cPickle.load(f)

                epoch = d['epoch']
                if epoch in measures[activation]: # Skip this epoch if its already been processed
                    continue                      # this is a trick to allow us to rerun this cell multiple times)
                    
                if epoch > MAX_EPOCHS:
                    continue
                
                num_layers = len(d['data']['activity_tst'])
                if PLOT_LAYERS is None:
                    PLOT_LAYERS = []
                    for lndx in range(num_layers):
                        PLOT_LAYERS.append(lndx)
                        Sx.append([])
                        Sy.append([])
                        
                cepochdata = defaultdict(list)
                for lndx in range(num_layers):
                    activity = d['data']['activity_tst'][lndx]
                    # Compute marginal entropies
                    h_upper = entropy_func_upper([activity,])[0]
                    hM_given_X = kde.kde_condentropy(activity, noise_variance)

                    # Compute conditional entropies of layer activity given output
                    hM_given_Y_upper=0.
                    for i in range(NUM_LABELS):
                        hcond_upper = entropy_func_upper([activity[saved_labelixs[i],:],])[0]
                        hM_given_Y_upper += labelprobs[i] * hcond_upper
                        
                    if DO_LOWER:
                        hM_given_Y_lower=0.
                        for i in range(NUM_LABELS):
                            hcond_lower = entropy_func_lower([activity[saved_labelixs[i],:],])[0]
                            hM_given_Y_lower += labelprobs[i] * hcond_lower
                        
                    cepochdata['MI_XM_upper'].append( nats2bits * (h_upper - hM_given_X) )
                    cepochdata['MI_YM_upper'].append( nats2bits * (h_upper - hM_given_Y_upper) )
                    cepochdata['H_M_upper'  ].append( nats2bits * h_upper )

                    pstr = 'upper: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (cepochdata['MI_XM_upper'][-1], cepochdata['MI_YM_upper'][-1])
                    
                    if(do_type_2 and activation == "linear"):
                        Sx[lndx].append(cepochdata['MI_XM_upper'][-1])
                        Sy[lndx].append(cepochdata['MI_YM_upper'][-1])

                    print('- Layer %d %s' % (lndx, pstr))

                measures[activation][epoch] = cepochdata

        if(do_type_1):
            plot_type_1(measures, PLOT_LAYERS, COLORBAR_MAX_EPOCHS, infoplane_measure, name_output_1)
        if(do_type_2):
            plot_type_2(Sx, Sy, name_output_2)
            # plot_type_2(measures, PLOT_LAYERS, COLORBAR_MAX_EPOCHS, infoplane_measure, name_output_2)