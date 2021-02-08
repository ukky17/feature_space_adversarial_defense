exp_name=resnet50_test

log=results/${exp_name}/log.txt
mkdir results/${exp_name}
CUDA_VISIBLE_DEVICES=0 python attack.py --dataset stl10 \
                                        --basenet ResNet50_stl \
                                        --model_path model_pth/resnet50_stl.pth \
                                        --exp_name ${exp_name} \
                                        --target_layer layer3 \
                                        --epochs 1500 \
                                        --lambda_c 0.00025 \
                                        --lambda_adv 0.00025 \
                                        --lambda_smooth 0.01 \
                                        --sigma 0.15 \
                                        > ${log}
