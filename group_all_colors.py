# group all the segments of all colours into one single segmentation file
# this script requires 3D-slicer and should be placed in the same directory where 3D-slicer is installed
# please refer to the documentation of 3D-slicer for more details

# imports
import slicer
import os
import shutil

# functions
def load_model(path, color):
    bone_model = slicer.util.loadModel(path)
    bone_model.GetDisplayNode().SetColor(
        color_vals[color][0], color_vals[color][1], color_vals[color][2]
    )
    bone_model.GetDisplayNode().SetVisibility(False)
    return bone_model


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
    else:
        raise ImportError("Unsucessful import of model to segmentation")
    slicer.mrmlScene.RemoveNode(slicer.util.getNode(model.GetName()))
    return segmentation_node


def export_segmentation_to_model(segmentation_node, model_folder_name):
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    exportFolderItemId = shNode.CreateFolderItem(
        shNode.GetSceneItemID(), model_folder_name
    )
    slicer.modules.segmentations.logic().ExportAllSegmentsToModels(
        segmentation_node, exportFolderItemId
    )
    # slicer.mrmlScene.RemoveNode(segmentation_node)
    print("Segmentation exported to Model!")


def save_model(filepath, color):
    slicer.util.saveNode(slicer.util.getFirstNodeByName(color + "s"), filepath)
    print("Model saved as stl file!")


def group_colors(segmentation_node, stl_files, color):
    seg_edit_wid = slicer.qMRMLSegmentEditorWidget()
    seg_edit_wid.setMRMLScene(slicer.mrmlScene)
    seg_edit_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    seg_edit_wid.setMRMLSegmentEditorNode(seg_edit_node)
    seg_edit_wid.setSegmentationNode(segmentation_node)

    seg_edit_wid.setActiveEffectByName("Logical operators")
    effect = seg_edit_wid.activeEffect()
    effect.setParameter("Operation", "UNION")

    segment_name = color + "s"
    segmentation_node.GetSegmentation().AddEmptySegment(segment_name)
    segmentation_node.GetSegmentation().GetSegment(segment_name).SetColor(
        color_vals[color][0], color_vals[color][1], color_vals[color][2]
    )

    target_seg_id = segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(
        segment_name
    )
    seg_edit_node.SetSelectedSegmentID(target_seg_id)

    for i in stl_files:
        effect.setParameter(
            "ModifierSegmentID",
            segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(
                i.split(".")[0]
            ),
        )
        effect.self().onApply()
        segmentation_node.GetSegmentation().RemoveSegment(i.split(".")[0])

    return segmentation_node


def clear_console():
    slicer.app.pythonConsole().clear()


def clear_scene():
    slicer.mrmlScene.Clear(0)


# main
def main():
    """

	Performs the combination of all the segments for all the colors and from the corresponding folder of the sample, 
	removes the individual segments and generates the single segmentation file in the nifti format and copies it to the specified folder

	Inputs:
		none, as all the colors (RGBY) are considered and processed to their corresponding segmentation nifti file

	Outputs:
		returns nothing, however removes all the individual segments for all the colors and generates the final combined segmentation in the given folder
		updates the user accordingly with clarifying print statements

	"""
    # clear_console()
    clear_scene()
    slicer.util.selectModule("Data")
    default_path = "D:\\process\\test"
    ctscan_folders = [
        i
        for i in os.listdir(default_path)
        if (not i.endswith("DS_Store")) and (not i.endswith("csv"))
    ]
    global color_vals
    color_vals = {
        "blue": (0, 0, 1),
        "green": (0, 1, 0),
        "red": (1, 0, 0),
        "yellow": (1, 1, 0),
    }
    colors = ["blue", "green", "red", "yellow"]

    for folder_num in ctscan_folders:

        for color in colors:

            clear_scene()
            print(f"Running CT_scan & Color: {folder_num}-{color}")
            # files and filepaths
            ctvol_path = os.path.join(default_path, folder_num)
            stl_files = [
                i for i in os.listdir(ctvol_path) if i.endswith(".stl") and color in i
            ]

            if len(stl_files) == 0:
                print(f"No STL files present for {color}")

            elif len(stl_files) == 1:
                print(f"Single STL file present for {color}")
                src_name = os.path.join(default_path, folder_num, stl_files[0])
                tgt_name = os.path.join(
                    default_path, folder_num, (folder_num + color + ".stl")
                )
                os.rename(src_name, tgt_name)

            else:
                print(f"Multiple STL files present for color: {color}")
                print(f"Grouping all {color} STL files ... ")

                stl_paths = [os.path.join(ctvol_path, i) for i in stl_files]
                # processing
                models = [load_model(i, color) for i in stl_paths]
                ct_segmentation_name = folder_num + color
                ct_segmentation = create_new_segmentation(ct_segmentation_name)
                for i in models:
                    ct_segmentation = import_model_to_segmentation(i, ct_segmentation)
                ct_segmentation.GetDisplayNode().SetVisibility(False)
                ct_segmentation = group_colors(ct_segmentation, stl_files, color)
                export_segmentation_to_model(ct_segmentation, ct_segmentation_name)
                save_model(
                    os.path.join(ctvol_path, ct_segmentation_name + ".stl"), color
                )

                for i in stl_paths:
                    os.remove(i)
                print(f"Success: {folder_num}-{color}!")
                print("------------------------\n")

        src_dir = ctvol_path
        tgt_dir = "D:\\process\\grouping_done_ct_scans\\"
        shutil.move(src_dir, tgt_dir)
        print(f"\nFinished {folder_num}!")
        print("------------------------------------------------------------\n")


if __name__ == "__main__":
    main()

