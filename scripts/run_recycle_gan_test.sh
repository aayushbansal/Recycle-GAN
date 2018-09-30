#!./scripts/train_recycle.sh
python test.py --dataroot ./datasets/ --name  --model cycle_gan  --which_model_netG resnet_6blocks   --dataset_mode unaligned  --no_dropout --gpu 1  --how_many 100  --loadSize 256 
