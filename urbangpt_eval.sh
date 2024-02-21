# to fill in the following path to evaluation!
output_model=./tw2t_multi_reg-cla-gird
datapath=./data/NYC_taxi/NYC_taxi.json
st_data_path=./data/NYC_taxi/NYC_taxi_pkl.pkl
res_path=./result_test/cross-region/NYC_taxi
start_id=0
end_id=51920
num_gpus=8

python ./urbangpt/eval/run_urbangpt.py --model-name ${output_model}  --prompting_file ${datapath} --st_data_path ${st_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}