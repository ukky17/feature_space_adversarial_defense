filename=201203_3

log=generated/${filename}/log_layer.txt
mkdir generated/${filename}
CUDA_VISIBLE_DEVICES=0 python attack_onestep_D_woCohen.py --filename ${filename} \
                                                          --target_layer -1 \
                                                          --epochs 1500 \
                                                          --lambda_c 0.001 \
                                                          --lambda_adv 0.001 > ${log}
