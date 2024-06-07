import json
import pickle
import os
import numpy as np

from dataloader import get_dataloader
import argparse
from dataloader import get_pretrain_task_batch
import re

# =============================== Setting =============================== #
args = argparse.ArgumentParser(prefix_chars='--', description='test')
# NYC_multi(for train)     NYC_taxi NYC_bike NYC_crime1 NYC_crime2 CHI_taxi (for test)
args.add_argument('-dataset_name', default='NYC_crime2', type=str)
# Only one option can be set to True
args.add_argument('-for_zeroshot', default=True, type=eval, help='for zero-shot prediction or not')
args.add_argument('-for_supervised', default=False, type=eval, help='for supervised prediction or not')
args.add_argument('-for_ablation', default=False, type=eval, help='for ablation study or not')

args.add_argument('-his', default=12, type=int)
args.add_argument('-pre', default=12, type=int)
args.add_argument('-batch_size', default=1, type=int)
args.add_argument('-input_base_dim', default=2, type=int)
args.add_argument('-input_extra_dim', default=5, type=int)
args.add_argument('-part_of_region', default=False, type=eval)
args.add_argument('-region_start', default=0, type=int)
args.add_argument('-region_end', default=80, type=int)
args = args.parse_args()

if args.dataset_name == 'NYC_multi':
    args.for_test = False
    args.json_path = args.dataset_name + '.json'
    args.pkl_path = args.dataset_name + '_pkl.pkl'
else:
    args.for_test = True
    if args.for_zeroshot:
        args.json_path = args.dataset_name + '_zeroshot.json'
        args.pkl_path = args.dataset_name + '_zeroshot_pkl.pkl'
    elif args.for_supervised:
        args.json_path = args.dataset_name + '_supervised.json'
        args.pkl_path = args.dataset_name + '_supervised_pkl.pkl'
    elif args.for_ablation:
        args.json_path = args.dataset_name + '_ablation.json'
        args.pkl_path = args.dataset_name + '_ablation_pkl.pkl'
    else:
        args.json_path = args.dataset_name + '.json'
        args.pkl_path = args.dataset_name + '_pkl.pkl'

if args.for_test:
    args.shuffle = False
else:
    args.shuffle = True
# =============================== Temporal Instructions =============================== #
time_ori_list = []
time_ori_list_5m = []
time_ori_list_60m = []
for i in range(1, 49):
    hours = (i - 1) // 2
    minutes = (i - 1) % 2 * 30
    time_str = f"{hours:02d}:{minutes:02d}"
    time_ori_list.append(time_str)
for i in range(1, 7):
    hours = (i - 1) * 4
    minutes = 0
    time_str = f"{hours:02d}:{minutes:02d}"
    time_ori_list_60m.append(time_str)
for i in range(1, 289):
    hours = (i - 1) // 12
    minutes = (i - 1) % 12 * 5
    time_str = f"{hours:02d}:{minutes:02d}"
    time_ori_list_5m.append(time_str)
week_ori_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
month_ori_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                  'August', 'September', 'October', 'November', 'December']

def time_decode(data, args, type):
    month_start_index = int(data[:, 0, 0, args.input_base_dim + 3])
    month_end_index = int(data[:, -1, 0, args.input_base_dim + 3])
    day_start_index = int(data[:, 0, 0, args.input_base_dim + 2])
    day_end_index = int(data[:, -1, 0, args.input_base_dim + 2])
    year_start_index = int(data[:, 0, 0, args.input_base_dim + 4])
    year_end_index = int(data[:, -1, 0, args.input_base_dim + 4])
    time_start_index = int(data[:, 0, 0, args.input_base_dim])
    time_end_index = int(data[:, -1, 0, args.input_base_dim])
    week_start_index = int(data[:, 0, 0, args.input_base_dim + 1])
    week_end_index = int(data[:, -1, 0, args.input_base_dim + 1])

    month_start, month_end = month_ori_list[month_start_index-1], month_ori_list[month_end_index-1]
    day_start, day_end = day_start_index, day_end_index
    year_start, year_end = year_start_index, year_end_index
    if type == 3 or type == 4 or type == 6:
        time_start, time_end = time_ori_list_60m[time_start_index - 1], time_ori_list_60m[time_end_index - 1]
    elif type == 1 or type == 2 or type == 5:
        time_start, time_end = time_ori_list[time_start_index-1], time_ori_list[time_end_index-1]
    else:
        time_start, time_end = time_ori_list_5m[time_start_index - 1], time_ori_list_5m[time_end_index - 1]
    week_start, week_end = week_ori_list[week_start_index-1], week_ori_list[week_end_index-1]

    if type == 3 or type == 4 or type == 6:
        interval = str(1) + "-day intervals'"
    elif type == 1 or type == 2 or type == 5:
        interval = str(30) + "-minute intervals'"
    else:
        interval = str(5) + "-minute intervals'"
    # interval_str = "prediction points" if isPred else "data points"
    interval_str = "data points"
    time_return = "'" + month_start + " " + str(day_start) + ", " + str(year_start) + ", " + \
                  time_start + ", " + week_start + " to " + month_end + " " + str(day_end) + ", " + \
                  str(year_end) + ", " + time_end + ", " + week_end + ", with " + interval_str + \
                  " recorded at " + interval
    return time_return

# =============================== Spatial Instructions =============================== #
def region_decode_ori(region_idx, type):
    if type == 1:
        granularity = 'within a three-kilometer radius'
    else:
        granularity = 'within a one-kilometer radius'
    pois_categ_list = []
    region_index_info = region_json[str(region_idx)]
    if len(region_index_info) != 0:
        borough_name = region_index_info[0]['borough_name']
        for poi_index in region_index_info:
            pois_categ_list.append(poi_index['category_name'])
        pois_categ_list = list(set(pois_categ_list))
        pois_categ_str = str(pois_categ_list)[1:-1].replace("'", "")
        region_return = " This region is located within the " + borough_name + " borough district and " \
                         "encompasses various POIs " + granularity + ", covering " + pois_categ_str + \
                         " categories. "
    else:
        region_return = " No description is available for this region. "
    return region_return

def region_decode_others(region_idx, type, region_json_others):
    if type == 5:
        granularity = 'within a four-kilometer radius'
    else:
        granularity = 'within a one-kilometer radius'
    pois_categ_list = []
    region_index_info = region_json_others[region_idx]
    if len(region_index_info["name"]) != 0:
        city_name_list = region_index_info["vicinity"]
        for string in city_name_list:
            if ',' in string:
                after_comma = string.split(',', 1)[1].strip()
                city_name = after_comma
                break
        if 'city_name' not in locals():
            city_name = city_name_list[0]

        pois_categ_list = region_index_info["types"]
        pois_set = set(pois_categ_list)
        pois_set.discard('locality')
        pois_set.discard('point_of_interest')
        pois_categ_list = list(pois_set)[:10]
        pois_categ_str = str(pois_categ_list)[1:-1].replace('"', '').replace("'", "")
        region_return = " This region is located within the city of " + city_name + " and " \
                         "encompasses various POIs " + granularity + ", covering " + pois_categ_str + \
                         " categories. "
    else:
        region_return = " No description is available for this region. "
    return region_return

def region_decode(region_start, region_end):
    region_return = ""
    for i in range(region_end - region_start):
        pois_categ_list = []
        region_index_info = region_json[str(region_start+i)]
        if len(region_index_info) != 0:
            borough_name = region_index_info[0]['borough_name']
            for poi_index in region_index_info:
                pois_categ_list.append(poi_index['category_name'])
            pois_categ_list = list(set(pois_categ_list))
            pois_categ_str = str(pois_categ_list)[1:-1].replace("'", "")
            region_return = region_return + str(i+1) + ". This region is located within the " + borough_name + " borough district and " \
                             "encompasses various POIs, including those belonging to the " + pois_categ_str + \
                             " categories. "
        else:
            region_return = region_return + str(i+1) + ". No description is available for this region. "
    return region_return

def zone_decode(zone_start, zone_end):
    region_return = ""
    for i in range(zone_end-zone_start):
        pois_categ_list = []
        region_index_info = zone_json[str(zone_start+i+1)]
        if len(region_index_info) != 0:
            borough_name = region_index_info[0]['borough_name']
            for poi_index in region_index_info:
                pois_categ_list.append(poi_index['category_name'])
            pois_categ_list = list(set(pois_categ_list))
            pois_categ_str = str(pois_categ_list)[1:-1].replace("'", "")
            region_return = region_return + str(i+1) + ". This region is located within the " + borough_name + " borough district and " \
                             "encompasses various POIs, including those belonging to the " + pois_categ_str + \
                             " categories. "
        else:
            region_return = region_return + str(i+1) + ". No description is available for this region. "
    return region_return

def data_zone_decode(data_gpt, zone_start, zone_end):
    region_return = ""
    for i in range(zone_end-zone_start):
        pois_categ_list = []
        region_index_info = zone_json[str(zone_start+i+1)]
        data_inflow = data_gpt[0, :, i, 0].tolist()
        data_outflow = data_gpt[0, :, i, 1].tolist()
        if len(region_index_info) != 0:
            borough_name = region_index_info[0]['borough_name']
            for poi_index in region_index_info:
                pois_categ_list.append(poi_index['category_name'])
            pois_categ_list = list(set(pois_categ_list))
            pois_categ_str = str(pois_categ_list)[1:-1].replace("'", "")
            region_return = region_return + str(i+1) + ". The historical inflow and outflow data are " + \
                            str(data_inflow) + " and " + str(data_outflow) + ", respectively. " \
                            "Additionally, the region is located within the " + borough_name + " borough district and " \
                            "encompasses various POIs, including those belonging to the " + pois_categ_str + " categories. "
        else:
            region_return = region_return + str(i+1) + ". The historical inflow and outflow data are " + \
                            str(data_inflow) + " and " + str(data_outflow) + ", respectively. " \
                            "No description is available for this region. "
    return region_return



list_all = []
data_all = []


# for NYC_BIKE AND NYC_CRIME
with open('st_data/poi/region_poi.json') as f:
    region_json = json.load(f)
# for CHI_TAXI
with open(f'./st_data/poi/CHI_taxi_POIs.json') as f:
    region_json_CHI_taxi = json.load(f)
# for NYC_TAXI
with open('st_data/poi/zone_poi.json') as f:
    zone_json = json.load(f)

# =============================== data Generation =============================== #
x_trn, y_trn = get_dataloader(args)
spt_x, spt_y, train_len = get_pretrain_task_batch(args, x_trn, y_trn, shuffle=args.shuffle)
for i in range(train_len):
    data, label = spt_x[i], spt_y[i]
    print(i, train_len)
    # generate st_data_all
    if args.part_of_region:
        data = data[:, :, args.region_start:args.region_end, :]
        label = label[:, :, args.region_start:args.region_end, :]
    dict_data = {}
    dict_data["data_x"], dict_data["data_y"] = np.concatenate([data[..., :args.input_base_dim], data[..., -1:]], axis=-1), \
                                               np.concatenate([label[..., :args.input_base_dim], label[..., -1:]], axis=-1)
    data_all.append(dict_data)

    region_nums = data.shape[2]
    for region_index in range(0, data.shape[2], 1):
        region_start = region_index
        region_end = region_index + 1
        if region_end > (region_nums - 1):
            region_end = region_nums
        list_conversations = []
        dict_main = {}
        dict_conversation_human = {}
        dict_conversation_gpt = {}

        list_gpt_datain = []
        list_gpt_lblsin = []
        data_gpt = data[:, :, region_start:region_end, :]
        label_gpt = label[:, :, region_start:region_end, :]
        for dim_index in range(args.input_base_dim):
            list_gpt_datain.append(data_gpt[0, :, 0, dim_index].astype(int))
            list_gpt_lblsin.append(label_gpt[0, :, 0, dim_index].astype(int))

        # =============================== Format Standardization =============================== #
        str_inflow = str(list_gpt_datain[0]).replace(",", "")
        str_inflow = re.sub(r'\s+', ' ', str_inflow)
        if str_inflow[1] == " ":
            str_inflow = str_inflow[:1] + str_inflow[2:]
        if args.input_base_dim > 1:
            str_outflow = str(list_gpt_datain[1]).replace(",", "")
            str_outflow = re.sub(r'\s+', ' ', str_outflow)
            if str_outflow[1] == " ":
                str_outflow = str_outflow[:1] + str_outflow[2:]

        lbls_inflow = str(list_gpt_lblsin[0]).replace(",", "")
        lbls_outflow = str(list_gpt_lblsin[1]).replace(",", "")
        lbls_inflow = re.sub(r'\s+', ' ', lbls_inflow)
        lbls_outflow = re.sub(r'\s+', ' ', lbls_outflow)
        if lbls_inflow[1] == " ":
            lbls_inflow = lbls_inflow[:1] + lbls_inflow[2:]
        if lbls_outflow[1] == " ":
            lbls_outflow = lbls_outflow[:1] + lbls_outflow[2:]


        # =============================== Instruction Generated =============================== #
        type = data_gpt[0, 0, 0, -1]
        region_index_new = region_index + args.region_start
        if type == 1:
            value_of_human = "Given the historical data for taxi flow over 12 time steps in a specific region of New York City, " \
                             "the recorded taxi inflows are " + str_inflow + ", and the recorded taxi outflows are " + str_outflow + \
                             ". The recording time of the historical data is " + time_decode(data_gpt, args, type) + \
                             ". Here is the region information:" + region_decode_ori(region_index_new, type) + "Now we want to predict " \
                             "the taxi inflow and outflow for the next 12 time steps during the time period of " + \
                             time_decode(label_gpt, args, type) + ". To improve prediction accuracy, a spatio-temporal " \
                             "model is utilized to encode the historical taxi data as tokens <ST_HIS>, where the first and the second tokens " \
                             "correspond to the representations of taxi inflow and outflow. Please conduct " \
                             "an analysis of the traffic patterns in this region, taking into account the provided time and regional " \
                             "information, and then generate the predictive tokens for regression, in the form \"<ST_PRE>\"."

            value_of_gpt = "Based on the given information, the predictive tokens of taxi inflow and outflow in this region " \
                           "are <ST_PRE>."
        elif type == 2:
            value_of_human = "Given the historical data for bike flow over 12 time steps in a specific region of New York City, " \
                             "the recorded bike inflows are " + str_inflow + ", and the recorded bike outflows are " + str_outflow + \
                             ". The recording time of the historical data is " + time_decode(data_gpt, args, type) + \
                             ". Here is the region information:" + region_decode_ori(region_index_new, type) + "We now aim to predict " \
                             "the bike inflow and outflow for the next 12 time steps during the time period of " + \
                             time_decode(label_gpt, args, type) + ". To improve prediction accuracy, a spatio-temporal " \
                             "model is utilized to encode the historical bike data as tokens <ST_HIS>, where the first and the second tokens " \
                             "correspond to the representations of bike inflow and outflow. Please conduct " \
                             "an analysis of the traffic patterns in this region, taking into account the provided time and regional " \
                             "information, and then generate the predictive tokens for regression, in the form \"<ST_PRE>\"."

            value_of_gpt = "Based on the given information, the predictive tokens of bike inflow and outflow in this region " \
                           "are <ST_PRE>."
        elif type == 3:
            value_of_human = "Given the historical data for crime over 12 time steps in a specific region of New York City, " \
                             "the recorded number of burglaries is " + str_inflow + ", and the recorded number of larcenies is " + str_outflow + \
                             ". The recording time of the historical data is " + time_decode(data_gpt, args, type) + \
                             ". Here is the region information:" + region_decode_ori(region_index_new, type) + "Now we aim to predict " \
                             "whether the two specific crimes will occur in this region within the next 12 time steps during the time period of " + \
                             time_decode(label_gpt, args, type) + ". To improve prediction accuracy, a spatio-temporal " \
                             "model is utilized to encode the historical crime data as tokens <ST_HIS>, where the first and the second tokens " \
                             "correspond to the representations of burglaries and larcenies. Please conduct " \
                             "an analysis of the crime patterns in this region, considering the provided time and regional " \
                             "information, and then generate the prediction tokens for classification, in the form \"<ST_PRE>\"."

            value_of_gpt = "Based on the given information, the predicted tokens of crime in this region are <ST_PRE>."

        elif type == 4:
            value_of_human = "Given the historical data for crime over 12 time steps in a specific region of New York City, " \
                             "the recorded number of robberies is " + str_inflow + ", and the recorded number of assaults is " + str_outflow + \
                             ". The recording time of the historical data is " + time_decode(data_gpt, args, type) + \
                             ". Here is the region information:" + region_decode_ori(region_index_new, type) + "Now we aim to predict " \
                             "whether the two specific crimes will occur in this region within the next 12 time steps during the time period of " + \
                             time_decode(label_gpt, args, type) + ". To improve prediction accuracy, a spatio-temporal " \
                             "model is utilized to encode the historical crime data as tokens <ST_HIS>, where the first and the second tokens " \
                             "correspond to the representations of robberies and assaults. Please conduct " \
                             "an analysis of the crime patterns in this region, considering the provided time and regional " \
                             "information, and then generate the prediction tokens for classification, in the form \"<ST_PRE>\"."

            value_of_gpt = "Based on the given information, the predicted tokens of crime in this region " \
                           "are <ST_PRE>."

        elif type == 5:
            value_of_human = "Given the historical data for taxi flow over 12 time steps in a specific region of Chicago, " \
                             "the recorded taxi inflows are " + str_inflow + ", and the recorded taxi outflows are " + str_outflow + \
                             ". The recording time of the historical data is " + time_decode(data_gpt, args, type) + \
                             ". Here is the region information:" + region_decode_others(region_index_new, type, region_json_CHI_taxi) + "Now we want to predict " \
                             "the taxi inflow and outflow for the next 12 time steps during the time period of " + \
                             time_decode(label_gpt, args, type) + ". To improve prediction accuracy, a spatio-temporal " \
                             "model is utilized to encode the historical taxi data as tokens <ST_HIS>, where the first and the second tokens " \
                             "correspond to the representations of taxi inflow and outflow. Please conduct " \
                             "an analysis of the traffic patterns in this region, taking into account the provided time and regional " \
                             "information, and then generate the predictive tokens for regression, in the form \"<ST_PRE>\"."

            value_of_gpt = "Based on the given information, the predictive tokens of taxi inflow and outflow in this region " \
                           "are <ST_PRE>."

        dict_main["id"] = 'train_' + args.dataset_name + '_region_' + str(region_start) + '_' + str(region_end) + '_len_' + str(i)
        dict_conversation_human["from"], dict_conversation_human["value"] = "human", value_of_human
        dict_conversation_gpt["from"], dict_conversation_gpt["value"] = "gpt", value_of_gpt
        list_conversations.append(dict_conversation_human)
        list_conversations.append(dict_conversation_gpt)
        dict_main["conversations"] = list_conversations
        list_all.append(dict_main)

# =============================== .json and .pkl Saved =============================== #
folder_path = './generated_file'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' was created.")
json_savepath = os.path.join(folder_path, args.json_path)
b = json.dumps(list_all)
f2 = open(json_savepath, 'w')
f2.write(b)
b=None
f2.close()
pkl_savepath = os.path.join(folder_path, args.pkl_path)
with open(pkl_savepath, 'wb') as file:
    pickle.dump(data_all, file)