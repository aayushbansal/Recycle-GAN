#!./scripts/train_recycle.sh
python train.py --dataroot ./datasets/<DATASET-NAME> --name <NAME> --model reCycle_gan  --which_model_netG resnet_6blocks --which_model_netP unet_128 --npf 8  --dataset_mode unaligned_triplet  --no_dropout --gpu 3 --pool_size 0 
