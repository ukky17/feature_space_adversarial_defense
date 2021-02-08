filename=201208_7

log=results/${filename}/log.txt
mkdir results/${filename}
CUDA_VISIBLE_DEVICES=0 python attack.py --dataset stl10 \
                                        --basenet ResNet50_stl \
                                        --model_path model_path/resnet50_stl.pth \
                                        --filename ${filename} \
                                        --target_layer layer3 \
                                        --epochs 1500 \
                                        --lambda_c 0.00025 \
                                        --lambda_adv 0.00025 \
                                        --lambda_smooth 0.05 \
                                        --sigma 0.15 \
                                        > ${log}
