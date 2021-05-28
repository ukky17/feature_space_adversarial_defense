exp_name=vgg16_test

log=results/${exp_name}/log.txt
mkdir results/${exp_name}
CUDA_VISIBLE_DEVICES=0 python attack.py --dataset stl10 \
                                        --basenet VGG_stl \
                                        --model_path model_pth/vgg_stl.pth \
                                        --exp_name ${exp_name} \
                                        --target_layer 20 \
                                        --epochs 1500 \
                                        --lambda_c 0.25 \
                                        --lambda_adv 0.25 \
                                        --lambda_smooth 0.5 \
                                        --sigma 0.15 \
                                        > ${log}
