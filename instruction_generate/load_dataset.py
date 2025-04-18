import os
import numpy as np
import pandas as pd

def time_add(data, dataset_name, week_start, month_day_start, month_start, interval=5, weekday_only=False, year_list=None, holiday_list=None, month_list_spe=None, day_start=0, hour_of_day=24):
    # day and week
    if weekday_only:
        week_max = 5
    else:
        week_max = 7
    month_max = 12
    time_slot = hour_of_day * 60 // interval
    day_data = np.zeros_like(data[..., :1])
    week_data = np.zeros_like(data[..., :1])
    month_day_data = np.zeros_like(data[..., :1])
    month_data = np.zeros_like(data[..., :1])
    year_data = np.zeros_like(data[..., :1])
    type_data = np.zeros_like(data[..., :1])
    # holiday_data = np.zeros_like(data[..., :1])

    # index_data = np.zeros_like(data)
    day_init = day_start
    week_init = week_start
    month_day_init = month_day_start
    month_init = month_start
    year_init  = year_list[0]
    year_start = year_list[0]
    year_init_tem = 0
    month_init_tem = 0
    year_dy_list = []
    month_dm_list = []

    month_index = 0
    year_max = year_list[-1]
    month_plus_index = 0
    year_plus_index = 0
    # holiday_init = 1


    month_yeap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_others = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    for year_idx in year_list:
        if year_idx % 4 == 0:
            year_temporary = 366
            month_dm_list.append(month_yeap)
        else:
            year_temporary = 365
            month_dm_list.append(month_others)
        year_dy_list.append(year_temporary)


    print('year_dy_list', year_dy_list)

    if len(data.shape) == 3:
        len_data = data.shape[1]
    elif len(data.shape) == 4:
        len_data = data.shape[2]
    else:
        raise ValueError

    for index in range(len_data):
        # add time_of_day
        if (index) % time_slot == 0:
            day_init = day_start
        day_init = day_init + 1

        # add day_of_week
        if (index) % time_slot == 0 and index !=0:
            week_init = week_init + 1
            if month_list_spe is not None:
                if week_init > week_max:
                    month_day_init = month_day_init + 3
                else:
                    month_day_init = month_day_init + 1
            else:
                month_day_init = month_day_init + 1
        if week_init > week_max:
            week_init = 1
        if year_init % 4 == 0:
            month_day_max = month_yeap[month_init - 1]
        else:
            month_day_max = month_others[month_init - 1]
        if month_day_init > month_day_max:
            month_day_init = 1

        # add index_of_year
        if year_init != year_init_tem:
            year_plus_index = year_plus_index + year_dy_list[year_init-year_start]
            year_init_tem = year_init
        if (index) % (time_slot * year_plus_index) == 0 and index != 0:
            year_init = year_init + 1

        # add index_of_month
        if month_init != month_init_tem:
            if month_list_spe is not None:
                month_plus_index = month_plus_index + month_list_spe[month_init-5]
                month_init_tem = month_init
            else:
                month_plus_index = month_plus_index + month_dm_list[year_init-year_start][month_init-1]
                month_init_tem = month_init
        if (index) % (time_slot*month_plus_index) == 0 and index != 0:
            month_init = month_init + 1
        if month_init > month_max:
            month_init = 1

        # add holiday
        if day_init < 6:
            holiday_init = 1
        else:
            holiday_init = 2

        day_data[..., index:index + 1, :] = day_init
        week_data[..., index:index + 1, :] = week_init
        month_day_data[..., index:index + 1, :] = month_day_init
        month_data[..., index:index + 1, :] = month_init
        year_data[..., index:index + 1, :] = year_init


    if dataset_name == 'NYCtaxi':
        type_data = type_data + 1
    elif dataset_name == 'NYCbike':
        type_data = type_data + 2
    elif dataset_name == 'NYCcrime1':
        type_data = type_data + 3
    elif dataset_name == 'NYCcrime2':
        type_data = type_data + 4
    elif dataset_name == 'CHItaxi':
        type_data = type_data + 5
    elif dataset_name == 'CHI_crime':
        type_data = type_data + 6
    else:
        raise ValueError

    return day_data, week_data, month_day_data, month_data, year_data, type_data
    # return day_data, week_data, month_day_data, month_data, year_data, holiday_data

def load_st_dataset(dataset, args):
    # Data recorded: 2016.01.01---2021.12.31
    if dataset == 'NYCmulti':
        # NYC_TAXI for training
        data_taxi_path = os.path.join('st_data/all_nyc_taxi_263x105216x2.npz')
        data_taxi = np.load(data_taxi_path)['data']
        # print(data.shape, data[data==0].shape)
        month_start = 1
        week_start = 5
        month_day_start = 1
        holiday_list = None
        interval = 30
        week_day = 7
        month = 12
        year_list = [2016, 2017, 2018, 2019, 2020, 2021]
        args.interval = interval
        args.week_day = week_day
        args.month=month
        day_data, week_data, month_day_data, month_data, year_data, type_data = time_add(
            data_taxi, 'NYCtaxi', week_start, month_day_start, month_start, interval=interval, weekday_only=False, year_list = year_list,
            holiday_list=holiday_list)
        data_taxi = np.concatenate([data_taxi, day_data, week_data, month_day_data, month_data, year_data, type_data], axis=-1)
        data_taxi = data_taxi[:80, (366) * (24 * 60 // 30):(366 + (31 + 28 + 31)) * (24 * 60 // 30), :]

        # NYC_BIKE for training
        selected_regions = np.load('./st_data/selected_regions.npz')['arr_0']
        interval = 30
        data_bike_path = os.path.join('st_data/all_nyc_bike_46x47x105216x2.npz')
        data_bike = np.load(data_bike_path)['data']
        day_data, week_data, month_day_data, month_data, year_data, type_data = time_add(
            data_bike, 'NYCbike', week_start, month_day_start, month_start, interval=interval, weekday_only=False, year_list = year_list,
            holiday_list=holiday_list)
        data_bike = np.concatenate([data_bike, day_data, week_data, month_day_data, month_data, year_data, type_data], axis=-1)
        # data_bike = data_bike[:, :, (366 + 31) * (24 * 60 // 30):(366 + (31 + 28)) * (24 * 60 // 30), :]
        data_bike = data_bike[:, :, (366 + 31 + 28 + 31) * (24 * 60 // 30):(366 + (31 + 28 + 31 + 30 + 31 + 30)) * (24 * 60 // 30), :]
        data_bike = data_bike.reshape(-1, 91*48, 8)[selected_regions[:80], ...]
        look_bike = data_bike[..., :2]
        print(look_bike.shape, look_bike[look_bike == 0].shape, look_bike[look_bike > 0].shape)

        # NYC_CRIME for training
        interval = 1440
        data_crime_path = os.path.join('st_data/crime_nyc_2016_1_2021_12_46x47.npz')
        data_crime1 = np.load(data_crime_path)['data'][..., 0:2]
        day_data, week_data, month_day_data, month_data, year_data, type_data = time_add(
            data_crime1, 'NYCcrime1', week_start, month_day_start, month_start, interval=interval, weekday_only=False, year_list = year_list,
            holiday_list=holiday_list)
        data_crime1 = np.concatenate([data_crime1, day_data, week_data, month_day_data, month_data, year_data, type_data], axis=-1)
        data_crime1 = data_crime1[:, :, (0) * ((24 * 60) // (24 * 60)):(366 + 365 + 365) * ((24 * 60) // (24 * 60)), :]
        data_crime1 = data_crime1.reshape(-1, 366 + 365 + 365, 8)[selected_regions[:80], ...]

        data_crime2 = np.load(data_crime_path)['data'][..., 2:4]
        day_data, week_data, month_day_data, month_data, year_data, type_data = time_add(
            data_crime2, 'NYCcrime2', week_start, month_day_start, month_start, interval=interval, weekday_only=False, year_list = year_list,
            holiday_list=holiday_list)
        data_crime2 = np.concatenate([data_crime2, day_data, week_data, month_day_data, month_data, year_data, type_data], axis=-1)
        data_crime2 = data_crime2[:, :, (0) * ((24 * 60) // (24 * 60)):(366 + 365 + 365) * ((24 * 60) // (24 * 60)), :]
        data_crime2 = data_crime2.reshape(-1, 366 + 365 + 365, 8)[selected_regions[:80], ...]

        data = np.concatenate([data_taxi, data_bike, data_crime1, data_crime2], axis=1)
        print('NYC_multi data shape:', data.shape, data_taxi.shape, data_bike.shape, data_crime1.shape, data_crime2.shape)

    elif dataset == 'NYCtaxi':
        data_taxi_path = os.path.join('st_data/all_nyc_taxi_263x105216x2.npz')
        data_taxi = np.load(data_taxi_path)['data']
        month_start = 1
        week_start = 5
        month_day_start = 1
        holiday_list = None
        interval = 30
        week_day = 7
        month = 12
        year_list = [2016, 2017, 2018, 2019, 2020, 2021]
        args.interval = interval
        args.week_day = week_day
        args.month=month
        day_data, week_data, month_day_data, month_data, year_data, type_data = time_add(
            data_taxi, 'NYCtaxi', week_start, month_day_start, month_start, interval=interval, weekday_only=False, year_list = year_list,
            holiday_list=holiday_list)
        data_taxi = np.concatenate([data_taxi, day_data, week_data, month_day_data, month_data, year_data, type_data], axis=-1)

        if args.for_zeroshot:
            data_taxi = data_taxi[80:160, (366 + 365 + 365 + 365) * (24 * 60 // 30):(366 + 365 + 365 + 365 + 14) * (24 * 60 // 30), :]
        elif args.for_supervised:
            data_taxi = data_taxi[:80, (-31) * (24 * 60 // 30):, :]
        elif args.for_ablation:
            data_taxi = data_taxi[80:160, (366) * (24 * 60 // 30):(366 + (31 + 28 + 31)) * (24 * 60 // 30), :]
        else:
            raise ValueError

        # =================================== use for Robustness study =================================== #
        # mask1, mask2, mask3, mask4 = getstd(data_taxi[..., :2])
        # print(mask1.shape, mask2.shape, mask3.shape, mask4.shape)
        # np.savez('NYC_taxi_mask1', data=mask1)
        # np.savez('NYC_taxi_mask2', data=mask2)
        # np.savez('NYC_taxi_mask3', data=mask3)
        # np.savez('NYC_taxi_mask4', data=mask4)
        data = data_taxi

    elif dataset == 'NYCbike':
        data_path = os.path.join('st_data/all_nyc_bike_46x47x105216x2.npz')
        data_bike = np.load(data_path)['data']  # only traffic speed data
        print(data_bike.shape, data_bike[data_bike==0].shape)
        month_start = 1
        month_day_start = 1
        week_start = 5
        holiday_list = None
        interval = 30
        week_day = 7
        month = 12
        year_list = [2016, 2017, 2018, 2019, 2020, 2021]
        args.interval = interval
        args.week_day = week_day
        args.month=month

        selected_regions = np.load('./st_data/selected_regions.npz')['arr_0']
        day_data, week_data, month_day_data, month_data, year_data, type_data = time_add(
            data_bike, 'NYCbike', week_start, month_day_start, month_start, interval=interval, weekday_only=False, year_list = year_list,
            holiday_list=holiday_list)
        data_bike = np.concatenate([data_bike, day_data, week_data, month_day_data, month_data, year_data, type_data], axis=-1)
        if args.for_zeroshot:
            data_bike = data_bike[:, :, (366 + 365 + 365 + 365) * (24 * 60 // 30):(366 + 365 + 365 + 365 +14) * (24 * 60 // 30), :]
            data_bike = data_bike.reshape(-1, 14*48, 8)[selected_regions[80:160], ...]
        elif args.for_supervised:
            data_bike = data_bike[:, :, (-31) * (24 * 60 // 30):, :]
            data_bike = data_bike.reshape(-1, 31*48, 8)[selected_regions[0:80], ...]
        elif args.for_ablation:
            data_bike = data_bike[:, :, (366 + 31 + 28 + 31) * (24 * 60 // 30):(366 + (31 + 28 + 31 + 30 + 31 + 30)) * (24 * 60 // 30), :]
            data_bike = data_bike.reshape(-1, 91*48, 8)[selected_regions[80:160], ...]
        else:
            raise ValueError
        data = data_bike

    elif dataset == 'NYCcrime1':
        data_path = os.path.join('st_data/crime_nyc_2016_1_2021_12_46x47.npz')
        data_crime1 = np.load(data_path)['data']  # DROP & PICK
        print(data_crime1.shape, data_crime1[data_crime1==0].shape)
        month_start = 1
        month_day_start = 1
        week_start = 5
        holiday_list = None
        interval = 1440
        week_day = 7
        month = 12
        year_list = [2016, 2017, 2018, 2019, 2020, 2021]
        args.interval = interval
        args.week_day = week_day
        args.month=month

        selected_regions = np.load('./st_data/selected_regions.npz')['arr_0']
        interval = 1440
        data_crime_path = os.path.join('st_data/crime_nyc_2016_1_2021_12_46x47.npz')
        data_crime1 = np.load(data_crime_path)['data'][..., 0:2]
        day_data, week_data, month_day_data, month_data, year_data, type_data = time_add(
            data_crime1, 'NYCcrime1', week_start, month_day_start, month_start, interval=interval, weekday_only=False, year_list = year_list,
            holiday_list=holiday_list)
        data_crime1 = np.concatenate([data_crime1, day_data, week_data, month_day_data, month_data, year_data, type_data], axis=-1)

        if args.for_zeroshot:
            data_crime1 = data_crime1[:, :, (366 + 365 + 365 + 365) * ((24 * 60) // (24 * 60)):(366 + 365 + 365 + 365 + 366) * ((24 * 60) // (24 * 60)), :]
            data_crime1 = data_crime1.reshape(-1, 366, 8)[selected_regions[80:160], ...]
        else:
            raise ValueError

        data = data_crime1

    elif dataset == 'NYCcrime2':
        data_path = os.path.join('st_data/crime_nyc_2016_1_2021_12_46x47.npz')
        data_crime1 = np.load(data_path)['data']
        print(data_crime1.shape, data_crime1[data_crime1==0].shape)
        month_start = 1
        month_day_start = 1
        week_start = 5
        holiday_list = None
        interval = 1440
        week_day = 7
        month = 12
        year_list = [2016, 2017, 2018, 2019, 2020, 2021]
        args.interval = interval
        args.week_day = week_day
        args.month=month

        selected_regions = np.load('./st_data/selected_regions.npz')['arr_0']
        interval = 1440
        data_crime_path = os.path.join('st_data/crime_nyc_2016_1_2021_12_46x47.npz')
        data_crime2 = np.load(data_crime_path)['data'][..., 2:4]
        day_data, week_data, month_day_data, month_data, year_data, type_data = time_add(
            data_crime2, 'NYCcrime2', week_start, month_day_start, month_start, interval=interval, weekday_only=False, year_list = year_list,
            holiday_list=holiday_list)
        data_crime2 = np.concatenate([data_crime2, day_data, week_data, month_day_data, month_data, year_data, type_data], axis=-1)

        if args.for_zeroshot:
            data_crime2 = data_crime2[:, :, (366 + 365 + 365 + 365) * ((24 * 60) // (24 * 60)):(366 + 365 + 365 + 365 + 366) * ((24 * 60) // (24 * 60)), :]
            data_crime2 = data_crime2.reshape(-1, 366, 8)[selected_regions[80:160], ...]
        else:
            raise ValueError

        data = data_crime2

    elif dataset == 'CHItaxi':
        data_taxi_path = os.path.join('./st_data/2021-CHI_taxi_77x17520x2.npz')
        data_taxi = np.load(data_taxi_path)['data']  # only the first dimension, traffic flow data
        print(data_taxi.shape, data_taxi[data_taxi==0].shape, max(data_taxi.reshape(-1)))
        # print(sss)
        month_start = 1
        week_start = 5
        month_day_start = 1
        holiday_list = None
        interval = 30
        week_day = 7
        month = 12
        year_list = [2021]
        args.interval = interval
        args.week_day = week_day
        args.month=month

        day_data, week_data, month_day_data, month_data, year_data, type_data = time_add(
            data_taxi, 'CHItaxi', week_start, month_day_start, month_start, interval=interval, weekday_only=False, year_list = year_list,
            holiday_list=holiday_list)
        data_taxi = np.concatenate([data_taxi, day_data, week_data, month_day_data, month_data, year_data, type_data], axis=-1)

        if args.for_zeroshot:
            data_taxi = data_taxi[:, (-31) * ((24 * 60) // interval):, :]
        else:
            raise ValueError

        # =================================== use for Robustness study =================================== #
        # mask1, mask2, mask3, mask4 = getstd(data_taxi[..., :2])
        # print(mask1.shape, mask2.shape, mask3.shape, mask4.shape)
        # np.savez('CHI_taxi_mask1', data=mask1)
        # np.savez('CHI_taxi_mask2', data=mask2)
        # np.savez('CHI_taxi_mask3', data=mask3)
        # np.savez('CHI_taxi_mask4', data=mask4)
        data = data_taxi
    elif dataset == 'CA_D5':
        data_tflow_path = os.path.join('./st_data/CA_D5.npz')
        data_tflow = np.load(data_tflow_path)['data']  # only the first dimension, traffic flow data
        print(data_tflow.shape)
        data_tflow = data_tflow.transpose(1, 0)
        data_tflow = np.expand_dims(data_tflow, axis=-1)
        data_tflow = np.concatenate([data_tflow, data_tflow], axis=-1)
        print(data_tflow.shape)
        # print(sss)
        month_start = 1
        week_start = 7
        month_day_start = 1
        holiday_list = None
        interval = 5
        week_day = 7
        month = 12
        year_list = [2017]
        args.interval = interval
        args.week_day = week_day
        args.month = month

        day_data, week_data, month_day_data, month_data, year_data, type_data = time_add(
            data_tflow, 'CA_D5', week_start, month_day_start, month_start, interval=interval, weekday_only=False,
            year_list=year_list,
            holiday_list=holiday_list)
        data_tflow = np.concatenate([data_tflow, day_data, week_data, month_day_data, month_data, year_data, type_data],
                                   axis=-1)
        data_tflow = data_tflow[:80, (0) * ((24 * 60) // interval):(7) * ((24 * 60) // interval), :]
        data = data_tflow
    elif dataset == 'PEMS07M':
        data_tspeed_path = os.path.join('./st_data/PeMS07M.csv')
        df_speed = pd.read_csv(data_tspeed_path, encoding='utf-8', header=None)
        data_tspeed = df_speed.values
        print(data_tspeed.shape)
        data_tspeed = data_tspeed.transpose(1, 0)
        data_tspeed = np.expand_dims(data_tspeed, axis=-1)
        data_tspeed = np.concatenate([data_tspeed, data_tspeed], axis=-1)
        print(data_tspeed.shape)
        # print(sss)
        month_start = 5
        week_start = 2
        month_day_start = 1
        holiday_list = None
        interval = 5
        week_day = 7
        month = 12
        year_list = [2012]
        month_list_spe = [23, 21]
        args.interval = interval
        args.week_day = week_day
        args.month = month

        day_data, week_data, month_day_data, month_data, year_data, type_data = time_add(
            data_tspeed, 'PEMS07M', week_start, month_day_start, month_start, interval=interval, weekday_only=True,
            year_list=year_list,
            holiday_list=holiday_list, month_list_spe=month_list_spe)
        data_tspeed = np.concatenate([data_tspeed, day_data, week_data, month_day_data, month_data, year_data, type_data],
                                    axis=-1)
        data_tspeed = data_tspeed[:80, (0) * ((24 * 60) // interval):(4) * ((24 * 60) // interval), :]
        data = data_tspeed
    elif dataset == 'CA_D11':
        data_tflow_path = os.path.join('./st_data/CA_D11.npz')
        data_tflow = np.load(data_tflow_path)['data']  # only the first dimension, traffic flow data
        print(data_tflow.shape)
        data_tflow = data_tflow.transpose(1, 0)
        data_tflow = np.expand_dims(data_tflow, axis=-1)
        data_tflow = np.concatenate([data_tflow, data_tflow], axis=-1)
        print(data_tflow.shape)
        # print(sss)
        month_start = 11
        week_start = 3
        month_day_start = 1
        holiday_list = None
        interval = 5
        week_day = 7
        month = 12
        year_list = [2017]
        args.interval = interval
        args.week_day = week_day
        args.month = month

        day_data, week_data, month_day_data, month_data, year_data, type_data = time_add(
            data_tflow, 'CA_D11', week_start, month_day_start, month_start, interval=interval, weekday_only=False,
            year_list=year_list,
            holiday_list=holiday_list)
        data_tflow = np.concatenate([data_tflow, day_data, week_data, month_day_data, month_data, year_data, type_data],
                                   axis=-1)
        data_tflow = data_tflow[:30, (0) * ((24 * 60) // interval):(7) * ((24 * 60) // interval), :]
        data = data_tflow
    elif dataset == 'METR_LA':
        data_tspeed_path = os.path.join('./st_data/METR_LA.npz')
        data_tspeed = np.load(data_tspeed_path)['data']  # only the first dimension, traffic flow data
        print(data_tspeed.shape)
        data_tspeed = data_tspeed.transpose(1, 0)
        data_tspeed = np.expand_dims(data_tspeed, axis=-1)
        data_tspeed = np.concatenate([data_tspeed, data_tspeed], axis=-1)
        print(data_tspeed.shape)
        # print(sss)
        month_start = 3
        week_start = 4
        month_day_start = 1
        holiday_list = None
        interval = 5
        week_day = 7
        month = 12
        year_list = [2012]
        args.interval = interval
        args.week_day = week_day
        args.month = month

        day_data, week_data, month_day_data, month_data, year_data, type_data = time_add(
            data_tspeed, 'METR_LA', week_start, month_day_start, month_start, interval=interval, weekday_only=False,
            year_list=year_list,
            holiday_list=holiday_list)
        data_tflow = np.concatenate([data_tspeed, day_data, week_data, month_day_data, month_data, year_data, type_data],
                                   axis=-1)
        data_tflow = data_tflow[:80, (0) * ((24 * 60) // interval):(7) * ((24 * 60) // interval), :]
        data = data_tflow

    else:
        raise ValueError

    print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
          data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 1:2].max(), data[..., 1:2].min(),
          data[..., 1:2].mean(), np.median(data[..., 1:2]), data.dtype)
    print('time: ', data[..., 2:3].max(), data[..., 2:3].min())
    print('day: ', data[..., 4:5].max(), data[..., 4:5].min())
    print('week: ', data[..., 3:4].max(), data[..., 3:4].min())
    print('month: ', data[..., 5:6].max(), data[..., 5:6].min())
    print('year: ', data[..., 6:7].max(), data[..., 6:7].min())
    print('type: ', data[..., 7:8].max(), data[..., 7:8].min())
    return data
