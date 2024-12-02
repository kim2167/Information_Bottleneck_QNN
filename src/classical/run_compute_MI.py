import compute_MI
import multiprocessing as mp

if __name__ == '__main__':

    enable_multiprocessing = False

    list_learning_rate = [0.00001, 0.00003, 0.00005, 0.00008, 0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01, 0.03, 0.05, 0.08, 0.1]
    list_layer_dim = [[32, 48, 24, 8, 4], [32, 48, 24, 8, 2], [32, 64, 24, 8, 4], [16, 32, 24, 8, 4], [12, 24, 24, 8, 8], [12, 24, 24, 8, 4], [64,64,64,64,64]]
    list_g_noise = [0.005,0.01,0.03,0.05, 0.06, 0.08, 0.085, 0.1, 0.12, 0.15]
    ratio_train = 1.0

    list_arch_layer = []
    for arch_layer in list_layer_dim:
        list_arch_layer.append('-'.join(map(str, arch_layer)))

    size_data = 32
    base_path = "/Users/Desktop/results/"

    gen_dataset = False 
    do_type_1 = False
    do_type_2 = True

    list_input_set = []
    for learning_rate in list_learning_rate:
        for g_noise in list_g_noise:
            for arch_layer in list_arch_layer:
                list_input_set.append((learning_rate, g_noise, arch_layer, size_data, ratio_train, gen_dataset, base_path, do_type_1, do_type_2)) 

    if(enable_multiprocessing):
        numCpu = 5
        pool = mp.Pool(processes = numCpu)
        pool.map(compute_MI.compute_MI, list_input_set)
        pool.close()
        pool.join()
    else:
        for input_set in list_input_set:
            compute_MI.compute_MI(input_set)