for s in input hidden
do
  CUDA_VISIBLE_DEVICES=0 python smoothing_predict.py --basenet ResNet50_stl \
                                                     --model_path model_pth/resnet50_stl.pth \
                                                     --target_layer layer3 \
                                                     --exp_name resnet50_test \
                                                     --n_data 8000 \
                                                     --space ${s}
done
