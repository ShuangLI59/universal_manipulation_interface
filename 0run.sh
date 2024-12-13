# accelerate --num_processes 1 train.py \
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-name=train_diffusion_unet_timm_umi_workspace \
    task.dataset_path=/store/real/shuang/diffusion_policy/data/diffusion_policy/cup_in_the_lab.zarr.zip