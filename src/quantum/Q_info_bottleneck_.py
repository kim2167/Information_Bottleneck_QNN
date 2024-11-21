'''
Created on Aug 10, 2024
'''
import os
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from numpy import linalg as LA
from scipy.linalg import logm
import torch
import math
import random
import matplotlib.pyplot as plt
nqubits = 6
dev = qml.device("default.mixed", wires=nqubits)

def gen_dataset(size_dataset, name_dataset):

    basis_vector = np.zeros((size_dataset, size_dataset+1))
    for i in range(size_dataset):
        basis_vector[i,size_dataset-1-i] = 1
        if i %2 == 0:
            basis_vector[i,-1] = 0
        else:
            basis_vector[i,-1] = 1

    np.savetxt(name_dataset, basis_vector, fmt="%d")

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


def get_dataset(size_dataset, ratio_training):

    path_dataset = "./orthog"+ str(size_dataset) + ".txt"

    if(not os.path.isfile(path_dataset)):
        gen_dataset(size_dataset, path_dataset)

    basis_vector = np.zeros((size_dataset,size_dataset+1))
    for i in range(size_dataset):
        basis_vector[i,size_dataset-1-i] = 1
        if i %2 == 0:
            basis_vector[i,-1] = 0
        else:
            basis_vector[i,-1] = 1

    data = np.loadtxt(path_dataset)
    X = np.array(data[:, :-1], requires_grad=False)
    Y = np.array(data[:, -1], requires_grad=False)
    Y = Y * 2 - np.ones(len(Y))  # shift label from {0, 1} to {-1, 1}


    num_train = int(ratio_training * size_dataset)
    index = np.random.permutation(range(size_dataset))

    X_train = X[index[:num_train]]
    X_test = Y[index[num_train:]]
    Y_train = Y[index[:num_train]]
    Y_test = Y[index[num_train:]]

    return X_train, X_test, Y_train, Y_test, X, Y



### Variational Quantum Classifier
def layer(W):
    for i in range(nqubits):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)

    for i in range(nqubits - 1):
        qml.CNOT(wires=[i, i+1])

    for i in range(nqubits):
        qml.AmplitudeDamping(0.1, wires=i)
        
        
def statepreparation(x):
    qml.AmplitudeEmbedding(x, wires=range(nqubits),normalize=True)

@qml.qnode(dev, interface="autograd")
def circuit(weights, x):
    statepreparation(x)
    for W in weights:
        layer(W)
    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

def accuracy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)
    return loss

def ansatz(weights):
    for W in weights:
        layer(W)
        
def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)


@qml.qnode(dev)
def get_x_input(x):
    if LA.norm(x) != 0:
        statepreparation(x)

    return qml.state()

def get_X_ensemble(X):
    x_ens = []
    for x in X:
        if LA.norm(x) != 0:
            a = get_x_input(x)
            x_ens.append(a)

    return x_ens

### Mutual Information Calculation
@qml.qnode(dev)
def get_x_input(x):
    if LA.norm(x) != 0:
        statepreparation(x)

    return qml.state()

def get_X_ensemble(X):
    x_ens = []
    for x in X:
        if LA.norm(x) != 0:
            a = get_x_input(x)
            x_ens.append(a)

    return x_ens


def mutual_info_calc(X,Y, template, wires, dev, *args, **kwargs):
    """Mutual information calculation."""
    @qml.qnode(dev)
    def find_output_state(input_state):
        statepreparation(input_state)
        template(*args, **kwargs)
        return qml.state()

    # state dim
    num_qubit = len(wires)
    state_dim = 2**num_qubit
    tensor_dim = state_dim * state_dim
    xhaty_dim = 2* state_dim
    ystate_dim = 2

    # prepare input, output ensemble, target ensemble
    input_ensemble = get_X_ensemble(X)
    output_ensemble = [find_output_state(x) for x in X if LA.norm(x) != 0]
    len_input = len(input_ensemble)

    # target ensemble
    target_ensemble = []
    ytrial = np.zeros((ystate_dim))
    for i in range(len_input):
        if Y[i] == 1:
            ytrial[0] = 1
        else:
            ytrial[1] = 1
        target_ensemble.append(np.outer(ytrial, np.conj(ytrial)))
        ytrial = np.zeros((ystate_dim))

    # prepare the mixed state of x, xhat, y
    x_dmat = np.zeros((state_dim, state_dim))
    y_dmat = np.zeros((ystate_dim, ystate_dim))
    xhat_dmat = np.zeros((state_dim, state_dim))
    for i in range(len_input):
        x_dmat = x_dmat + input_ensemble[i]/len_input
        y_dmat = y_dmat + target_ensemble[i]/len_input
        xhat_dmat = xhat_dmat +  output_ensemble[i]/len_input

    # construct the joint state of the composite system x, xhat
    joint_dmat_xxhat = np.zeros((tensor_dim, tensor_dim))
    for i in range(len_input):
        tp_dmat_xxhat = np.kron(input_ensemble[i],output_ensemble[i])
        assert(tp_dmat_xxhat.shape==joint_dmat_xxhat.shape)
        joint_dmat_xxhat = joint_dmat_xxhat + tp_dmat_xxhat/len_input

    # construct the joint state of the composite system xhat, y
    joint_dmat_xhaty = np.zeros((xhaty_dim, xhaty_dim))
    for i in range(len_input):
        tp_dmat_xhaty = np.kron(output_ensemble[i],target_ensemble[i])
        assert(tp_dmat_xhaty.shape==joint_dmat_xhaty.shape)
        joint_dmat_xhaty = joint_dmat_xhaty + tp_dmat_xhaty/len_input

    # entropy of x, y, xhat
    Sx = -np.real(np.trace(np.matmul(x_dmat, logm(x_dmat))) / math.log(2))
    Sy = -np.real(np.trace(np.matmul(y_dmat, logm(y_dmat))) / math.log(2))
    Sxhat = -np.real(np.trace(np.matmul(xhat_dmat, logm(xhat_dmat))) / math.log(2))

    # entropy of joint states, Sxxhat, Sxhaty
    Sxxhat = -np.real(np.trace(np.matmul(joint_dmat_xxhat, logm(joint_dmat_xxhat)))/ math.log(2))
    Sxhaty = -np.real(np.trace(np.matmul(joint_dmat_xhaty, logm(joint_dmat_xhaty)))/ math.log(2))

    # relative entropy
    relative_Sx_xhat = -Sxxhat + Sx + Sxhat
    relative_Sxhat_y = -Sxhaty + Sy + Sxhat

    beta = 0.5
    L = beta * relative_Sx_xhat - (1-beta) * relative_Sxhat_y

    print(f"relative entropy S_x_xhat: {relative_Sx_xhat}")
    print(f"relative entropy S_xhat_y: {relative_Sxhat_y}")
    print(f"objective IB: {L}\n\n")

    return relative_Sx_xhat,relative_Sxhat_y,L

if __name__ == '__main__':
    
    size_data = 64
    ratio_train = 0.8

    X_train, X_test, Y_train, Y_test, X, Y = get_dataset(size_data, ratio_train) 
    num_qubits = nqubits
    num_layers = 6

    weights_init = 0.02 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
    bias_init = np.array(0.0, requires_grad=True)

    # You can use different optimizer also.
    # https://docs.pennylane.ai/en/stable/code/qml.html
    opt = NesterovMomentumOptimizer(0.0008)
    batch_size = 32

    losses = []
    vals = []

    Sx_1=[]
    Sy_1=[]
    mut_infos_1 = []

    Sx_2 = []
    Sy_2 = []
    mut_infos_2=[]

    Sx_3 = []
    Sy_3 = []
    mut_infos_3=[]

    Sx_4 = []
    Sy_4 = []
    mut_infos_4=[]

    Sx_5 = []
    Sy_5 = []
    mut_infos_5=[]

    Sx_6 = []
    Sy_6 = []
    mut_infos_6=[]

    # train the variational classifier
    weights = weights_init
    bias = bias_init
    for it in range(500):
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = X_train[batch_index]
        Y_batch = Y_train[batch_index]
        weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)

        # Compute accuracy
        predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X]
        acc = accuracy(Y, predictions)

        # compute mutual info
        # 1st layer
        S_xxhat_1,S_xhaty_1,mutual_1 = mutual_info_calc(X, Y, ansatz, range(nqubits), dev, weights[:1])
        Sx_1.append(S_xxhat_1)
        Sy_1.append(S_xhaty_1)
        mut_infos_1.append(mutual_1)

        # 2rd layer
        S_xxhat_2,S_xhaty_2,mutual_2 = mutual_info_calc(X, Y, ansatz, range(nqubits), dev, weights[:2])
        Sx_2.append(S_xxhat_2)
        Sy_2.append(S_xhaty_2)
        mut_infos_2.append(mutual_2)

        # 3rd layer
        S_xxhat_3,S_xhaty_3,mutual_3 = mutual_info_calc(X, Y, ansatz, range(nqubits), dev, weights[:3])
        Sx_3.append(S_xxhat_3)
        Sy_3.append(S_xhaty_3)
        mut_infos_3.append(mutual_3)

        # 4th layer
        S_xxhat_4,S_xhaty_4,mutual_4 = mutual_info_calc(X, Y, ansatz, range(nqubits), dev, weights[:4])
        Sx_4.append(S_xxhat_4)
        Sy_4.append(S_xhaty_4)
        mut_infos_4.append(mutual_4)

        # 5th layer
        S_xxhat_5,S_xhaty_5,mutual_5 = mutual_info_calc(X, Y, ansatz, range(nqubits), dev, weights[:5])
        Sx_5.append(S_xxhat_5)
        Sy_5.append(S_xhaty_5)
        mut_infos_5.append(mutual_5)

        #6 th layer
        S_xxhat_6,S_xhaty_6,mutual_6 = mutual_info_calc(X, Y, ansatz, range(nqubits), dev, weights[:6])
        Sx_6.append(S_xxhat_6)
        Sy_6.append(S_xhaty_6)
        mut_infos_6.append(mutual_6)

        losses.append(acc)

        print(
            "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Information: {:0.7f} "
            "".format(it + 1, cost(weights, bias, X, Y), acc, 0)
        )

        print(len(Sx_1))

    fig = plt.figure(figsize=(16,12))

    plt.scatter(Sx_1,Sy_1,color="red",label="layer1")
    plt.plot(Sx_1,Sy_1,color="red",linestyle='dashed')

    plt.scatter(Sx_2,Sy_2,color="orange",label="layer2")
    plt.plot(Sx_2,Sy_2,color="orange",linestyle='dashed')

    plt.scatter(Sx_3,Sy_3,color="blue",label="layer3")
    plt.plot(Sx_3,Sy_3,color="blue",linestyle='dashed')

    plt.scatter(Sx_4,Sy_4,color="green",label="layer4")
    plt.plot(Sx_4,Sy_4,color="green",linestyle='dashed')

    plt.scatter(Sx_5,Sy_5,color="purple",label="layer5")
    plt.plot(Sx_5,Sy_5,color="purple",linestyle='dashed')

    plt.scatter(Sx_6,Sy_6,color="black",label="layer6")
    plt.plot(Sx_6,Sy_6,color="black",linestyle='dashed')

    plt.legend(fontsize=15)
    plt.xlabel('I(X;T)', fontsize = 20) 
    plt.ylabel('I(Y;T)', fontsize = 20) 
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)

    plt.savefig("plot_1.png", dpi=300)
    plt.clf()
    plt.close()
    plt.plot(losses)
    plt.savefig("plot_2.png", dpi=300)
    plt.clf()
    plt.close()
