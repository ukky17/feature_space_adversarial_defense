filename=201209_3

log=generated/${filename}/log_layer.txt
mkdir generated/${filename}
CUDA_VISIBLE_DEVICES=0 python attack_onestep_D.py --filename ${filename} \
                                                  --target_layer 27 \
                                                  --epochs 1500 \
                                                  --lambda_c 0.25 \
                                                  --lambda_adv 0.25 \
                                                  --lambda_smooth 2.5 \
                                                  --cohen_sigma 0.15 \
                                                  > ${log}
