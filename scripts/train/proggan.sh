#!/bin/bash


# === Experiment configuration ===
gan_type="ProgGAN"
learn_alphas=false
learn_gammas=false
num_support_sets=200000
num_support_dipoles=512
min_shift_magnitude=0.1
max_shift_magnitude=6.0
reconstructor_type="ResNet"
batch_size=8
max_iter=200000
tensorboard=true
lambda_reg=0.25
flow="FFJORD"
hid_dim=5
reg_flow=1.0
small_dim="--small-dim"
flow_modules="512-512-512-512-512"
seed=0
# ================================


# Run training script
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
                --ckp-freq=50000 \
                --lambda-reg=${lambda_reg} \
                --flow=${flow} \
		            --reg-flow=${reg_flow} \
                $small_dim \
                --flow-modules=${flow_modules} \
                --seed=${seed}
