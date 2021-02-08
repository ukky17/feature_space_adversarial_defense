filename=201209_3

log=results/${filename}/log.txt
mkdir results/${filename}
CUDA_VISIBLE_DEVICES=0 python attack.py --dataset stl10 \
                                        --basenet VGG_stl \
                                        --model_path model_pth/vgg_stl.pth \
                                        --filename ${filename} \
                                        --target_layer 27 \
                                        --epochs 1500 \
                                        --lambda_c 0.25 \
                                        --lambda_adv 0.25 \
                                        --lambda_smooth 2.5 \
                                        --sigma 0.15 \
                                        > ${log}
