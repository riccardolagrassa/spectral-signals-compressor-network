import rasterio
import re
import numpy
import os
import shutil






def get_mean_and_std(dataloader,save_path):
    channels_sum, channels_squared_sum, max_range,min_range = 0, 0, 0, 0
    for data in dataloader:
        images_name = data.split('/')[-1]
        folder_name = data.split('/')[-2]
        r=rasterio.open(save_path+folder_name+'_'+images_name, "r")
        bands = [i for i in range(1, r.count + 1)]
        r = r.read(bands)
        r = r.astype('float32')

        for j in bands:
            tmp_min=numpy.nanmin(r[j-1])
            tmp_max=numpy.nanmax(r[j - 1])
            if tmp_min < min_range:
                min_range=tmp_min
            if tmp_max > max_range:
                max_range= tmp_max

        # Mean over batch, height and width, but not over the channels
        #channels_sum += r_data.mean(axis=(1, 2))
        #channels_squared_sum += numpy.mean(r_data**2, axis=(1,2))

        # max_range_tmp = numpy.max(r_data)
        # min_range_tmp = numpy.min(r_data)
        # if max_range_tmp > max_range:
        #     max_range = max_range_tmp
        # if min_range_tmp < min_range:
        #     min_range = min_range_tmp



    #mean = channels_sum/len(dataloader)
    # std = sqrt(E[X^2] - (E[X])^2)
    #std = (channels_squared_sum / len(dataloader) - mean ** 2) ** 0.5
    return max_range, min_range


def get_index_data_filtered():
    data_list=[]
    for (path, dirs, files) in os.walk(main_path):
        # print("Processing folder ", idx)
        for idx_file, name in enumerate(files):
            if re.match('[0-9]+.tif$', name):
                if len(data_list) == dataset_slice:
                    return data_list
                else:
                    data_list.append(path + '/' + name)


def get_data(mode_sampler, save_path):
        for name in mode_sampler:
            images_name = name.split('/')[-1]
            folder_name = name.split('/')[-2]
            #r=rasterio.open(save_path+folder_name+'_'+images_name)
            shutil.copyfile(name, save_path+folder_name+'_'+images_name)


dataset_slice=1000
path_save_file_train = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/train/'
path_save_file_test = '/home/super/datasets-nas/lombardia_original_slice_'+str(dataset_slice)+'/test/'

#
# if os.path.exists(path_save_file_train):
#     shutil.rmtree(path_save_file_train)
# os.makedirs(path_save_file_train)
#
# if os.path.exists(path_save_file_test):
#     shutil.rmtree(path_save_file_test)
# os.makedirs(path_save_file_test)



main_path='/home/super/ignazio/datasets/IREA/lombardia/data2017/'

#Get data index using filtering through regular expression
data_list=get_index_data_filtered()



#Train val splitting
validation_split=0.2
dataset_size = len(data_list)
indices = list(range(dataset_size))
split = int(numpy.floor(validation_split * dataset_size))
numpy.random.seed(1)
numpy.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = [data_list[i] for i in train_indices]
valid_sampler = [data_list[i] for i in val_indices]

#save train data
#get_data(train_sampler, path_save_file_train)
#save test data
#get_data(valid_sampler, path_save_file_test)


max_range, min_range=get_mean_and_std(train_sampler,path_save_file_train)
print(max_range, min_range)