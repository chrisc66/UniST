export DATASET="MciTRTDT5D2"
python main.py --device_id 0 --machine machine  --dataset $DATASET --task short --size middle --mask_strategy_random 'batch' --lr 3e-4 --used_data 'diverse'  --prompt_ST 0
