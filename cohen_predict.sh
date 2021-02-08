for i in 1 2
do
  for s in input hidden
  do
    CUDA_VISIBLE_DEVICES=0 python cohen_predict.py --basenet VGG_stl \
                                                   --model_path models/vgg_stl.pth \
                                                   --target_layer 27 \
                                                   --filename 201209_${i}_2_alldata \
                                                   --data_path $(find generated/201209_${i} -name \*variables.npy) \
                                                   --n_data 8000 \
                                                   --space ${s}
  done
done
