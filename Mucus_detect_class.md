# Mucus Algorithm

Mucus plug detection and classification pipeline.

## 1. Mucus plug one-stage detection

### Data preparation

Scripts under `utils/data_code/`:

- **make_patch_dataset.py**  
  Patch the images and labels.

- **make_txt.py**  
  Convert the mask annotations corresponding to each patch image to YOLO txt format.

- **make_kfold_yolo.py**  
  Split the dataset into 5 folds.

### Model training

- **train.py**  
  Train on 5-fold data sequentially.

### Model prediction

- **test_5fold.py**  
  Predict on 5-fold data.

### Model performance metrics

- **calculate_froc_patient.py**  
  Calculate patient-level FROC metrics.

## 2. Two-stage classification of mucus plugs

### Data preparation

Scripts under `Get_3D_Class_Data/`:

- **extract_mask_bboxes.py**  
  Extract 3D bounding box information for each labeled region from the 3D mucus plug mask file.

- **categorize_confidence.py**  
  Categorize the first-stage FP candidates by confidence level.

- **merge_gt_to_prediction.py**  
  Write the extracted GT detection bounding boxes to the corresponding Prediction CSV file by fold.

- **extract_patches.py**  
  Perform gray region filtering, stratified negative sampling, CT resampling, and patch cropping. TP rows are removed during gray region filtering. Final patches are saved as `.npy` files.

> See [Get_3D_Class_Data/3D分类数据制作.md](Get_3D_Class_Data/3D分类数据制作.md) for detailed step-by-step instructions (in Chinese).

### Model training

Scripts under `3D_resnet_class/`:

- **train_classifier.py**  
  Train on specified fold data.

- **dataset.py**  
  Data loading.

- **model.py**  
  3D classification model structure.

- **resnet.py**  
  MedicalNet 3D ResNet model backbone.

### Model prediction & metrics

- **eval_classifier_patient_level.py**  
  Classify the candidate bounding boxes from the first stage and calculate patient-level FROC.
