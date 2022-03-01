import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



def scatter_plot(rand_dist):
    for idx, j in enumerate(rand_dist):
        if idx == 3:
            break
        instances=np.moveaxis(j, -1, 0)
        r = stats.describe(instances[:,:,0])
        print(idx, "MinMax ", np.array(r[1]).mean(), "Mean ", np.array(r[2]).mean(), "var: ",np.array(r[3]).mean(), "Sk: ", np.array(r[4].mean()), "Kurt: ", np.array(r[5].mean()))
        plt.scatter(instances[:,0], instances[:, 1], s=1, label=idx)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()

def global_normalization(rand_dist):
    return (rand_dist - np.min(rand_dist)) / ( np.max(rand_dist) - np.min(rand_dist))

def local_normalization(rand_dist):
    for idx, instance in enumerate(rand_dist):
        rand_dist[idx] = (instance - np.min(instance)) / ( np.max(instance) - np.min(instance))
    return rand_dist

def per_channel_normalization(rand_dist):
    min_channel_list=[0 for i in range(9)]
    max_channel_list=[0 for i in range(9)]
    clone_transposed=(np.moveaxis(rand_dist, 0, -1))
    for idx, j in enumerate(clone_transposed):
        min_channel_list[idx] = np.min(j)
        max_channel_list[idx] = np.max(j)

    for idx, instance in enumerate(rand_dist):
        for idx1, channels in enumerate(instance):
            rand_dist[idx][idx1] = (channels - min_channel_list[idx1]) / (max_channel_list[idx1] - min_channel_list[idx1])
    return rand_dist


rand_dist = np.random.randn(400, 9, 40,40)
global_norm_dist = global_normalization(rand_dist.copy())
local_norm_dist = local_normalization(rand_dist.copy())
perchannel_normdist = per_channel_normalization(rand_dist.copy())

print("Global mean: ", np.mean(rand_dist), "Std: ",np.std(rand_dist), "Min: ", np.min(rand_dist), "Max: ", np.max(rand_dist), np.var(rand_dist))
print("Global norma mean: ", np.mean(global_norm_dist), "Std: ",np.std(global_norm_dist),"Min: ", np.min(global_norm_dist), "Max: ", np.max(global_norm_dist), np.var(global_norm_dist))
print("Local norma mean: ", np.mean(local_norm_dist), "Std: ", np.std(local_norm_dist),"Min: ", np.min(local_norm_dist), "Max: ", np.max(local_norm_dist), np.var(local_norm_dist))
print("Per channel norma mean: ", np.mean(perchannel_normdist), "Std: ", np.std(perchannel_normdist),"Min: ", np.min(perchannel_normdist), "Max: ", np.max(perchannel_normdist), np.var(perchannel_normdist))
scatter_plot(np.moveaxis(rand_dist, 0, -1))
scatter_plot(np.moveaxis(global_norm_dist, 0, -1))
scatter_plot(np.moveaxis(local_norm_dist, 0, -1))
scatter_plot(np.moveaxis(perchannel_normdist, 0, -1))