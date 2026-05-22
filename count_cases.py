import csv
import glob
import sys

dir_path = sys.argv[1] if len(sys.argv) > 1 else 'Get_3D_Class_Data'

tp_all = set()
fp_all = {}
for f in sorted(glob.glob(f'{dir_path}/Prediction_TP_FP_fold_*.csv')):
    fold = f.split('fold_')[1].replace('.csv', '')
    with open(f) as fh:
        reader = csv.DictReader(fh)
        tp_uids = set()
        fp_uids = set()
        for row in reader:
            uid = row['patient_key']
            if row['prediction_type'] == 'TP':
                tp_uids.add(uid)
            elif row['prediction_type'] == 'FP':
                fp_uids.add(uid)
    tp_all |= tp_uids
    fp_all[fold] = fp_uids
    print(f'fold_{fold}: TP={len(tp_uids)}, FP={len(fp_uids)}')

fp_union = set().union(*fp_all.values())
print(f'\nTP 总病例: {len(tp_all)}')
print(f'FP 总病例: {len(fp_union)}')

only_tp = tp_all - fp_union
only_fp = fp_union - tp_all

if only_tp:
    print(f'\n仅在TP中出现的 patient_key ({len(only_tp)}):')
    for uid in sorted(only_tp):
        print(f'  {uid}')
if only_fp:
    print(f'\n仅在FP中出现的 patient_key ({len(only_fp)}):')
    for uid in sorted(only_fp):
        print(f'  {uid}')
if not only_tp and not only_fp:
    print('\nTP与FP病例完全一致')
