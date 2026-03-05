# Mucus Detection Algorithm

This is a YOLO-based algorithm for detecting mucus plugs in DICOM images.

## Installation

```bash
chmod +x docker_install.sh
./docker_install.sh
```

> The current directory will be mounted into the container, so you can access data and result files in the same location on your host system.

## Usage

Run inside your Docker container:

```bash
cd source
python3 test_patch.py -i [DICOM_FOLDER_ROOT_PATH]
```

Example:
```bash
python3 test_patch.py -i /mnt/data/sources/src_mucus/send_mucusAlgorithms/data
```

## Checking Results

After execution, a `results.csv` file will be generated in the DICOM folder:

```bash
cat [DICOM_FOLDER_PATH]/results.csv
```

To generate visualization results:
```bash
python3 visualize_results.py -i [CSV_FILE_PATH]
```
Example:
```bash
python3 visualize_results.py -i ./results.csv
```

The resulting images will be saved in the `source/results/` folder.



