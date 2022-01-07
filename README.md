Deep learning based semantic segmentation of the humerus fractures, Author - Rajat Sharma

This repo consists of the code developed for the purpose of researching and developing a deep learning based method for semantic segmentation of the humerus fractures
the legend for the files present is as follows:

data_X (dir): ct-scans/volumes
data_Y (dir): masks (corresponding to the ct-scans)
unet3d.py: unet model (from scratch based on the Ronneberger unet paper)
unet_final.py (main): main code that runs the complete deep learning pipeline
preprocess.py: pre-processing functions of the ct scans and masks
postprocess.py: post-processing functions for the results
group_all_colors (3d-Slicer): using 3d-slicer groups all the different colours (one shot combination of all the segments)
group_by_color (3d-Slicer): similar to the above, but can be done for a single color at a time (user-input)
generate_seg (3d-Slicer): combines all the segments of different colours into that single color and generates the corresponding segmentation
organise_into_folders.py: organises the initial given data into proper folders consisting of STL files and ct-scans for easier working
sep_ct_seg.py: separates the ct scans and corresponding masks into data_X and data_Y folders (described above)


although not really a pre-requisite, however, it is advised to be a bit familiar with the open source software, 3d-slicer, would make it easier to actually understand what some of the above-mentioned 3d-slicer relevant scripts do. This is a major component of the thesis.
methodology:

the original data that was given comprised of the STL files and the ct-scan (nifti format)
40 samples were given with each sample consisting of STL files for different colors (RGBY) depending on which humerus bone segment is present
the segments corresponding to a color had to be combined into one segment and then this segment was converted to a segmentation file (nifti) (mask)
afterwards the deep learnign pipeline was established using Tensorflow and Keras
the architecture that was chosen was the famous Unet
for this, two approaches were adopted
first approach:

based on using a backbone with the choice of using pre-trained weights for the encoder part (transfer learning)
used the open source library - segmentation-models 3D which is a 3D version of the popular 2D-segmentation models and offers many options
extensive experimentation was carried out with respect to the hyperparameters and parameters


second approach:

more fundamental with the goal being to build the unet model based on the above paper from scratch (encoder-decoder architecture)
here too, several experiments were undertaken to assess and evaluate the model with different values for eg. learning rate etc.


preprocessing and postprocessing functions were implemented accordingly to the ct-scans and their masks
results were analysed and were reported



