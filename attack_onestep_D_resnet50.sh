filename=201208_7

log=generated/${filename}/log_layer.txt
mkdir generated/${filename}
CUDA_VISIBLE_DEVICES=0 python attack_onestep_D.py --dataset stl10 \
                                                  --basenet ResNet50_stl \
                                                  --model_path models/resnet50_stl.pth \
                                                  --filename ${filename} \
                                                  --target_layer layer3 \
                                                  --epochs 1500 \
                                                  --lambda_c 0.00025 \
                                                  --lambda_adv 0.00025 \
                                                  --lambda_smooth 0.05 \
                                                  --cohen_sigma 0.15 \
                                                  > ${log}
