# separates the ct-scans and corresponding masks from their respective folders to data_X and data_Y format

# imports
import shutil
import os

# organise the data
folders = os.listdir(combined_path)
for folder in folders:

    os.mkdir(os.path.join(x_path, folder))
    os.mkdir(os.path.join(y_path, folder))

    # ct_scan
    src_dir_ct = os.path.join(combined_path, folder, folder + "HRimage.nii.gz")
    tgt_dir_ct = os.path.join(x_path, folder, folder + "HRimage.nii.gz")
    shutil.move(src_dir_ct, tgt_dir_ct)

    # seg
    src_dir_seg = os.path.join(combined_path, folder, folder + "HRimage_seg.nii")
    tgt_dir_seg = os.path.join(y_path, folder, folder + "HRimage_seg.nii")
    shutil.move(src_dir_seg, tgt_dir_seg)

    os.rmdir(os.path.join(combined_path, folder))

    print(f"{folder} moved!")
