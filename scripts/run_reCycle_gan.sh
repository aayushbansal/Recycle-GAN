#!./scripts/train_recycle.sh
python train.py --dataroot ./datasets/<DATASET-NAME> --name <NAME> --model reCycle_gan  --which_model_netG resnet_6blocks --which_model_netP unet_128 --npf 8  --dataset_mode unaligned_triplet  --no_dropout --gpu 3 --identity 0  --pool_size 0 --save_latest_freq 10000 --niter 10 --niter_decay 10 --lr_decay_iters 5
