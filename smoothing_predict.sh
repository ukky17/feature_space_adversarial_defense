for s in input hidden
do
  CUDA_VISIBLE_DEVICES=0 python smoothing_predict.py --basenet VGG_stl \
                                                     --model_path model_pth/vgg_stl.pth \
                                                     --target_layer 27 \
                                                     --exp_name vgg16_test \
                                                     --n_data 8000 \
                                                     --space ${s}
done
