if [ true ]; then
    # make dataset(extract patch image and mask in the original image and mask)
    #python make_patch_dataset.py -input_path "/mnt/data/data/MuscusPlug_100_data_20250630_negative/image"
    #python make_patch_dataset.py -input_path "/mnt/data/data/MuscusPlug_100_data_20250630_positive/image"
    #python make_patch_dataset.py -input_path "/mnt/data/data/MuscusPlug_100_positive_20250917/image"
fi

if [ true ]; then
    RUN pip install connected-components-3d # it install only first turn
    python make_txt.py -path ../yolo_dataset -set train
fi
