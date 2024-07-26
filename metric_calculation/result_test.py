import torch
from metrics import All_Metrics
import json
import numpy as np
import os
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report

def test(mode, mae_thresh=None, mape_thresh=0.0):
    len_nums = 0
    y_pred_in = []
    y_true_in = []
    y_pred_out = []
    y_true_out = []

    y_true_in_regionlist = []
    y_pred_in_regionlist = []
    y_true_out_regionlist = []
    y_pred_out_regionlist = []
    index_all = 0

    # Retrieve all JSON files from a folder and sort them by filename
    file_list = sorted([filename for filename in os.listdir(folder_path) if filename.endswith(".json")])

    for idx, filename in enumerate(file_list):
        file_path = os.path.join(folder_path, filename)
        print(file_path)
        with open(file_path, "r") as file:
            data_t = json.load(file)

        for i in range(len(data_t)):
            i_data = data_t[i]
            y_in = np.array(i_data["y_in"])
            y_out = np.array(i_data["y_out"])
            st_pre_infolow = np.array(i_data["st_pre_infolow"])
            st_pre_outfolow = np.array(i_data["st_pre_outfolow"])
            i4data_all = int(data_t[i]["id"].split('_')[6])
            if index_all != i4data_all :
                len_nums = len_nums + 1
                y_true_in_region = np.stack(y_true_in, axis=-1)
                y_pred_in_region = np.stack(y_pred_in, axis=-1)
                y_true_out_region = np.stack(y_true_out, axis=-1)
                y_pred_out_region = np.stack(y_pred_out, axis=-1)
                y_true_in_regionlist.append(y_true_in_region)
                y_pred_in_regionlist.append(y_pred_in_region)
                y_true_out_regionlist.append(y_true_out_region)
                y_pred_out_regionlist.append(y_pred_out_region)
                y_pred_in = []
                y_true_in = []
                y_pred_out = []
                y_true_out = []
                index_all = i4data_all
            y_true_in.append(y_in)
            y_pred_in.append(st_pre_infolow)
            y_true_out.append(y_out)
            y_pred_out.append(st_pre_outfolow)
            if (i == len(data_t) - 1 and idx == len(file_list) - 1):
                y_true_in_region = np.stack(y_true_in, axis=-1)
                y_pred_in_region = np.stack(y_pred_in, axis=-1)
                y_true_out_region = np.stack(y_true_out, axis=-1)
                y_pred_out_region = np.stack(y_pred_out, axis=-1)
                y_true_in_regionlist.append(y_true_in_region)
                y_pred_in_regionlist.append(y_pred_in_region)
                y_true_out_regionlist.append(y_true_out_region)
                y_pred_out_regionlist.append(y_pred_out_region)
                y_pred_in = []
                y_true_in = []
                y_pred_out = []
                y_true_out = []
    print('len_nums', len_nums)
    y_true_in = np.stack(y_true_in_regionlist, axis=0)
    y_pred_in = np.stack(y_pred_in_regionlist, axis=0)
    y_true_out = np.stack(y_true_out_regionlist, axis=0)
    y_pred_out = np.stack(y_pred_out_regionlist, axis=0)
    y_pred_in, y_pred_out = np.abs(y_pred_in), np.abs(y_pred_out)
    print(y_true_in.shape, y_pred_in.shape, y_true_out.shape, y_pred_out.shape)

    if mode == 'classification':
        test_classfication(y_true_in, y_pred_in, y_true_out, y_pred_out)
    else:
        for t in range(y_true_in.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred_in[:, t, ...], y_true_in[:, t, ...], mae_thresh, mape_thresh, None)
            print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape * 100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred_in, y_true_in, mae_thresh, mape_thresh, None)
        print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape * 100))

        for t in range(y_true_in.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred_out[:, t, ...], y_true_out[:, t, ...], mae_thresh, mape_thresh, None)
            print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(t + 1, mae, rmse, mape * 100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred_out, y_true_out, mae_thresh, mape_thresh, None)
        print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(mae, rmse, mape * 100))


def test_classfication(y_true_in, y_pred_in, y_true_out, y_pred_out):

    for i in range(2):
        if i == 0:
            y_true = y_true_in
            y_pred = y_pred_in
        else:
            y_true = y_true_out
            y_pred = y_pred_out
        y_true[y_true > 1] = 1
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)

        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"MicroF1: {micro_f1:.2f}")
        print(f"MacroF1: {macro_f1:.2f}")
        print(f"f1 Score: {f1:.2f}")

################################ result path ################################
folder_path = 'result_test_file/tw2t_multi_reg-cla_NYC_taxi_final'
# folder_path = 'result_test_file/tw2t_multi_reg-cla_NYC_bike_final'

# 'BURGLARY': 0, 'GRAND LARCENY': 1, 'ROBBERY': 2, 'FELONY ASSAULT': 3
# folder_path = 'result_test_file/tw2t_multi_reg-cla_NYC_crime1_final'
# folder_path = 'result_test_file/tw2t_multi_reg-cla_NYC_crime2_final'

# mode = 'classification' # regression  or  classification
mode = 'regression'

# Make sure that the total length of your json file(s) a multiple of 80.
test(mode)
