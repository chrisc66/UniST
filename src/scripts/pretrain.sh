export DATASET="MciTRTDT5D2"
python main.py --device_id 0 --machine machine  --model_input_channels 3 --loss_channel_weights "[1.0,1.0,0.5]" --dataset $DATASET --task short --size middle --mask_strategy_random 'batch' --lr 3e-4 --used_data 'diverse'  --prompt_ST 0
