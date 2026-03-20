### count_image.py          统计数据目录中有多少个数据
### save_NiftiPths_csv.py   将dataset/nifti_files目录中各个图像niigz的地址写入csv文件中
### make_patch_dataset.py   根据数据目录中的nifti_image_paths.csv文件来对图片及mask数据进行裁剪并划分训练、验证及测试
### make_txt.py             将数据目录中的mask转换为符合yolo训练的txt格式
### make_kfold_yolo.py      对数据进行分折处理

## 处理过程
1.一份病例数据过来时,如果CT是dicom文件格式,首先将dicom转换为niigz文件
数据集格式命名
   -----data
        ----...image...
            ---patientID.niigz
            ---...
            ---...
        ----...mask...
            ---patientID.niigz
            ---...
            ---...
2.使用extract_nifti_paths.py来提取目录中的文件名,并将这些文件名放入到csv文件中
3.利用make_patch_dataset.py来将图片和标签进行分patch,最终将图片resize成[512,512,3](yxc)的图像,对应的mask标签数据为[1,512,512],并将总的数据分成train,valid,test
4.使用mask_text.py将patch图像对应的mask标签转换为txt文件(目标检测的标注)
5.使用make_kfold_yolo.py来将数据集划分成5折,每次训练使用其中一折
6.将划分的5折数据创建对应的yaml文件

上述完成后,数据集创建完成!!!