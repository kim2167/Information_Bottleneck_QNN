import numpy as np
import matplotlib.pyplot as plt

def transform_format_pharsed(result_pharse):

    transformed = []
    for epoch in range(0, len(result_pharse['X'])):
        transformed.append({'XM':result_pharse['X'][epoch], 'YM':result_pharse['Y'][epoch]})
        
    return transformed

def plot_type_1(result_pharse, PLOT_LAYERS, COLORBAR_MAX_EPOCHS, name_output):
    
    vals = transform_format_pharsed(result_pharse)
    sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))

    fig=plt.figure(figsize=(15,13))
    for epoch in range(0, len(vals)):
        c = sm.to_rgba(epoch)
        xmvals = np.array(vals[epoch]['XM'])[PLOT_LAYERS]
        ymvals = np.array(vals[epoch]['YM'])[PLOT_LAYERS]

        plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
        plt.scatter(xmvals, ymvals, s=20, facecolors=[c for _ in PLOT_LAYERS], edgecolor='none', zorder=2)

    plt.xlim([0, 6.1])
    plt.ylim([0, 1.1])
    plt.xlabel('I(X;M)')
    plt.ylabel('I(Y;M)')
    plt.title("tmp")
        
    cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8]) 
    plt.colorbar(sm, label='Epoch', cax=cbaxes)
    plt.tight_layout()
    plt.savefig(name_output, bbox_inches='tight')

    plt.clf()
    plt.close()

def line_handler(line_str):
    
    split_str = line_str.split(':')
    
    if(len(split_str) != 0):
    
        if(split_str[0].split(' ')[-1] == "S_x_xhat"):
            return 'X', float(split_str[-1])
        elif(split_str[0].split(' ')[-1] == "S_xhat_y"):
            return 'Y', float(split_str[-1])
        elif(split_str[0].split(' ')[-1] == "IB"):
            return 'IB', float(split_str[-1])

    return None, None

def pharse_txt(path_txt):
    
    # Opening file
    file_txt = open(path_txt, 'r')
    
    layer = 0
    result_tmp = {"X":[], 'Y':[], "IB":[]}
    result_pharse = {"X":[], 'Y':[], "IB":[]}
    for line in file_txt:
        # print(line)
        xy, num = line_handler(line)
        
        if(xy != None):
            result_tmp[xy].append(num)
            
            if(xy == "IB"):
                layer += 1
             
            if(layer == 6):
                for key in result_tmp:
                    result_pharse[key].append(result_tmp[key])
                
                result_tmp = {"X":[], 'Y':[], "IB":[]}
                layer = 0
                
    # print(len(result_pharse['X']))
            
    # Closing files
    file_txt.close()

    return result_pharse
    
if __name__ == '__main__':
    
    path_txt = "/Users/myeongsu/Desktop/32_30_0.2.txt"
    result_pharse = pharse_txt(path_txt)
    
    num_layers = 6
    num_epoch = 500
    
    COLORBAR_MAX_EPOCHS = num_epoch
    PLOT_LAYERS = []
    for i in range(0, num_layers):
        PLOT_LAYERS.append(i)
    
    path_output = path_txt.split('.')[0] + '.png'
    
    plot_type_1(result_pharse, PLOT_LAYERS, COLORBAR_MAX_EPOCHS, path_output)
    
    print("Done.")
    