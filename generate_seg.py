# generates the segmentation file in the nifti format for each color (if present) (RGBY) and saves it in the specified folder
# this script requires 3D-slicer and should be placed in the same directory where 3D-slicer is installed
# please refer to the documentation of 3D-slicer for more details

# imports
import slicer
import os
from natsort import natsorted
import shutil

# functions
def load_model(path):
    bone_model = slicer.util.loadModel(path)
    bone_model_name = bone_model.GetName()
    if "blue" in bone_model_name:
        bone_model.GetDisplayNode().SetColor(0, 0, 1)

    elif "green" in bone_model_name:
        bone_model.GetDisplayNode().SetColor(0, 1, 0)

    elif "red" in bone_model_name:
        bone_model.GetDisplayNode().SetColor(1, 0, 0)

    elif "yellow" in bone_model_name:
        bone_model.GetDisplayNode().SetColor(1, 1, 0)

    bone_model.GetDisplayNode().SetVisibility(False)
    return bone_model


def load_volume(path):
    ct_volume_node = slicer.util.loadVolume(path)
    return ct_volume_node


def create_new_segmentation(segmentation_name):
    new_segmentation_node = slicer.vtkMRMLSegmentationNode()
    new_segmentation_node.SetName(segmentation_name)
    slicer.mrmlScene.AddNode(new_segmentation_node)
    new_segmentation_node.CreateDefaultDisplayNodes()
    return new_segmentation_node


def import_model_to_segmentation(model, segmentation_node):
    import_bool = slicer.modules.segmentations.logic().ImportModelToSegmentationNode(
        slicer.util.getNode(model.GetName()), segmentation_node
    )
    if import_bool:
        print("Successful import of model to segmentation!")
        slicer.mrmlScene.RemoveNode(model)
    else:
        raise ImportError("Unsucessful import of model to segmentation")
    return segmentation_node


def export_to_labelmapvol(segmentation_node, ref_vol, labelmap_vol_name):
    new_labelmap_vol_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLLabelMapVolumeNode"
    )
    new_labelmap_vol_node.SetName(labelmap_vol_name)
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
        segmentation_node, new_labelmap_vol_node, ref_vol
    )
    print("Exported to labelmap volume!")
    return new_labelmap_vol_node


def save_node(labelmap_vol, save_filepath):
    slicer.util.saveNode(
        slicer.util.getNode(labelmap_vol.GetName()), (save_filepath + ".nii")
    )
    print("Labelmap volume saved!")


def clear_console():
    slicer.app.pythonConsole().clear()


def clear_scene():
    slicer.mrmlScene.Clear(0)


# main
def main():
    clear_console()
    clear_scene()

    slicer.util.selectModule("Data")
    default_path = "D:\\process\\test"
    ctscan_folders = natsorted(
        [
            i
            for i in os.listdir(default_path)
            if (not i.endswith("DS_Store")) and (not i.endswith("csv"))
        ]
    )
    for folder_num in ctscan_folders:
        print(f"Running CT_scan: {folder_num}")

        # files and filepaths
        ctvol_path = os.path.join(default_path, folder_num)
        ct_file = [i for i in os.listdir(ctvol_path) if i.endswith(".nii.gz")]
        ct_path = [os.path.join(ctvol_path, i) for i in ct_file]
        stl_files = natsorted([i for i in os.listdir(ctvol_path) if i.endswith(".stl")])
        stl_paths = natsorted([os.path.join(ctvol_path, i) for i in stl_files])

        # processing
        ct_vol_node = load_volume(ct_path[0])
        models = [load_model(i) for i in stl_paths]
        ct_segmentation_name = ct_file[0].split(".")[-3] + "_seg"
        ct_segmentation = create_new_segmentation(ct_segmentation_name)

        for i in models:
            ct_segmentation = import_model_to_segmentation(i, ct_segmentation)

        ct_segmentation.GetDisplayNode().SetVisibility(False)
        labelmap_vol_node = export_to_labelmapvol(
            ct_segmentation, ct_vol_node, ct_segmentation_name
        )
        save_node(
            labelmap_vol_node, os.path.join(ctvol_path, ct_segmentation_name + ".nii")
        )

        print(f"Success!\n")

        src_dir = ctvol_path
        tgt_dir = "D:\\process\\labelmap_vol_done\\"
        shutil.move(src_dir, tgt_dir)
        print(f"\nFinished: {folder_num}!")
        print("------------------------------------------------------------\n")


if __name__ == "__main__":
    main()

