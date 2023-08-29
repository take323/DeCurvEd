#!/bin/bash


# === Experiment configuration ===
gan_type="StyleGAN2"
stylegan2_resolution=1024
z_truncation=0.7
w_space=true
learn_alphas=false
learn_gammas=false
num_support_sets=200000
num_support_dipoles=512
min_shift_magnitude=0.1
max_shift_magnitude=6.0
reconstructor_type="ResNet"
batch_size=12
max_iter=200000
lambda_reg=0.25
tensorboard=true
flow="FFJORD"
reg_flow=1.0
small_dim="--small-dim"
flow_modules="512-512-512-512-512"
support_set_lr=0.0001
# ================================


# Run training script
shift_in_w_space=""
if $w_space ; then
  shift_in_w_space="--shift-in-w-space"
fi

learn_a=""
if $learn_alphas ; then
  learn_a="--learn-alphas"
fi

learn_g=""
if $learn_gammas ; then
  learn_g="--learn-gammas"
fi

tb=""
if $tensorboard ; then
  tb="--tensorboard"
fi

python train.py $tb \
                --gan-type=${gan_type} \
                --z-truncation=${z_truncation} \
                --stylegan2-resolution=${stylegan2_resolution} \
                $shift_in_w_space \
                --reconstructor-type=${reconstructor_type} \
                $learn_a \
                $learn_g \
                --num-support-sets=${num_support_sets} \
                --num-support-dipoles=${num_support_dipoles} \
                --min-shift-magnitude=${min_shift_magnitude} \
                --max-shift-magnitude=${max_shift_magnitude} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=100 \
                --ckp-freq=100000 \
		            --lambda-reg=${lambda_reg} \
                --flow=${flow} \
		            --reg-flow=${reg_flow} \
                $small_dim \
                --flow-modules=${flow_modules} \
                --support-set-lr=${support_set_lr}
