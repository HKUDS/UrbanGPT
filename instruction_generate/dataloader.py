import torch
import numpy as np
import torch.utils.data
from add_window import Add_Window_Horizon
from load_dataset import load_st_dataset
from normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
import random


def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=False):
    #load raw st dataset
    data = load_st_dataset(args.dataset_name, args)        # B, N, D
    data = data.transpose(1, 0, 2)

    # # normalize st data
    # data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    #
    # # spilit dataset by days or by ratio
    # if args.test_ratio > 1:
    #     data_train, data_val, data_test = split_data_by_days(data, args.val_ratio, args.test_ratio)
    # else:
    #     data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    # data_train = data
    # data = None

    #add time window
    if args.for_test:
        x_tra, y_tra = Add_Window_Horizon(data, args.his, args.pre, single)
    else:
        x_tra_taxi, y_tra_taxi = Add_Window_Horizon(data[:4320, ...], args.his, args.pre, single)
        x_tra_bike, y_tra_bike = Add_Window_Horizon(data[4320:4320 + 4368, ...], args.his, args.pre, single)
        x_tra_crime1, y_tra_crime1 = Add_Window_Horizon(data[4320 + 4368:4320 + 4368 + 1096, ...], args.his, args.pre,single)
        x_tra_crime2, y_tra_crime2 = Add_Window_Horizon(data[4320 + 4368 + 1096:4320 + 4368 + 1096 + 1096, ...], args.his, args.pre, single)
        x_tra = np.concatenate([x_tra_taxi, x_tra_bike, x_tra_crime1, x_tra_crime2], axis=0)
        y_tra = np.concatenate([y_tra_taxi, y_tra_bike, y_tra_crime1, y_tra_crime2], axis=0)

    # normalize st data
    # _, scaler_data, scaler_day, scaler_week, scaler_holiday = normalize_dataset(data, normalizer, args.input_base_dim)
    print('Train: ', x_tra.shape, y_tra.shape)
    return x_tra, y_tra

def get_pretrain_task_batch(args, x_tra, y_tra, shuffle=True):
    batch_size = args.batch_size
    len_dataset = x_tra.shape[0]

    batch_list_x = []
    batch_list_y = []
    permutation = np.random.permutation(len_dataset)
    for index in range(0, len_dataset, batch_size):
        start = index
        end = min(index + batch_size, len_dataset)
        indices = permutation[start:end]
        if shuffle:
            x_data = x_tra[indices.copy()]
            y_data = y_tra[indices.copy()]
        else:
            x_data = x_tra[start:end]
            y_data = y_tra[start:end]
        batch_list_x.append(x_data)
        batch_list_y.append(y_data)
    train_len = len(batch_list_x)
    return batch_list_x, batch_list_y, train_len


def normalize_dataset(data, normalizer, input_base_dim, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
            scaler = StandardScaler(mean, std)
            data[:, :, 0:input_base_dim] = scaler.transform(data[:, :, 0:input_base_dim])
        else:
            data_ori = data[:, :, 0:input_base_dim]
            data_day = data[:, :, input_base_dim:input_base_dim+1]
            data_week = data[:, :, input_base_dim+1:input_base_dim+2]

            mean_data = data_ori.mean()
            std_data = data_ori.std()
            mean_day = data_day.mean()
            std_day = data_day.std()
            mean_week = data_week.mean()
            std_week = data_week.std()

            scaler_data = StandardScaler(mean_data, std_data)
            data_ori = scaler_data.transform(data_ori)
            scaler_day = StandardScaler(mean_day, std_day)
            data_day = scaler_day.transform(data_day)
            scaler_week = StandardScaler(mean_week, std_week)
            data_week = scaler_week.transform(data_week)
            data = np.concatenate([data_ori, data_day, data_week], axis=-1)
            print(mean_data, std_data, mean_day, std_day, mean_week, std_week)
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler_data, scaler_day, scaler_week, None
    # return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data