import save_activations
import multiprocessing as mp

if __name__ == '__main__':

    base_path = "/Users/Desktop/results/"
    list_activation_function = ["tanh", "linear", "sigmoid"]
    # list_activation_function = ["linear"]
    list_learning_rate = [0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1]
    list_layer_dim = [[32, 48, 24, 8, 4], [32, 48, 24, 8, 2], [32, 64, 24, 8, 4], [16, 32, 24, 8, 4], [12, 24, 24, 8, 8]]
   
    batch_size = 32 
    ratio_train = 0.8
    num_epochs = 2000
    size_data = 64
    gen_dataset = False 

    list_input_set = []
    for activation_func in list_activation_function:
        for learning_rate in list_learning_rate:
            for layer_dim in list_layer_dim:
                list_input_set.append((activation_func, learning_rate, layer_dim, batch_size, num_epochs, ratio_train, base_path, size_data, gen_dataset))

                
    numCpu = int(mp.cpu_count()/2)
    pool = mp.Pool(processes = numCpu)
    pool.map(save_activations.save_activations, list_input_set)
    pool.close()
    pool.join()
