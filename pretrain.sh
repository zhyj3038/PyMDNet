python pretrain.py --result_dir train_models/otb_vot \
                   --dataset otb_vot \
                   --init_model_path models/init.npy

python pretrain.py --result_dir train_models/vot \
                   --dataset vot \
                   --init_model_path models/init.npy

python pretrain.py --result_dir train_models/otb \
                   --dataset otb \
                   --init_model_path models/init.npy
