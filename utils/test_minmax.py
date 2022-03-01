import numpy as np

min_channel_list=[0 for i in range(432)]
max_channel_list=[0 for i in range(432)]
samples = np.random.randn(432, 56, 70)
print(samples.shape)
for idx, j in enumerate(samples):

    min_channel_list[idx] = np.min(j)
    max_channel_list[idx] = np.max(j)

print(min_channel_list)
print(np.min(samples[1]))