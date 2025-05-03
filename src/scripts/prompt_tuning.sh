export DATASET="MciTRTDT5D2"
export PRETRAINED_MODEL_PATH="./experiments/Pretrain_Dataset_MciTRTDT5D2_Task_short_FewRatio_0.5/model_save/model_best"
python main.py --device_id 0 --machine machine --task short --size middle --prompt_ST 1  --pred_len 6 --his_len 6 --num_memory_spatial 512 --num_memory_temporal 512 --prompt_content 's_p_c'  --dataset $DATASET --lr 3e-4 --used_data 'diverse' --file_load_path $PRETRAINED_MODEL_PATH
