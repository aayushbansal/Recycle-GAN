#!./scripts/train_recycle.sh
python train.py --dataroot ./datasets/ --name <NAME> --model recycle_gan  --which_model_netG resnet_6blocks --which_model_netP unet_256 --dataset_mode unaligned_triplet  --no_dropout --gpu 2 --identity 0  --pool_size 0 
