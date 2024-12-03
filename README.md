# The Risk of Shortcutting in Deep Learning Algorithms in Medical Research

This repository contains the code to reproduce the experiments of the paper "he Risk of Shortcutting in Deep Learning 
Algorithms in Medical Research."

## Abstract
While deep learning (DL) offers the compelling ability to detect details beyond human vision, its black-box nature makes
it prone to misinterpretation. A key problem is algorithmic shortcutting, where DL models inform their predictions with
patterns in the data that are easy to detect algorithmically but potentially misleading. Shortcutting makes it trivial
to create models with surprisingly accurate predictions that lack all face validity. This case study shows how easily
shortcut learning happens, its danger, how complex it can be, and how hard it is to counter. We use simple ResNet18
convolutional neural networks (CNN) to train models to do two things they should not be able to do: predict which
patients avoid consuming refried beans or beer purely by examining their knee X-rays (AUC of 0.63 for refried beans and
0.74 for beer). We then show how these models’ abilities are tied to several confounding and latent variables in the
image. Moreover, the image features the models use to shortcut cannot merely be removed or adjusted through
pre-processing. The end result is that we must raise the threshold for evaluating research using CNNs to proclaim new 
medical attributes that are present in medical images.

# Code Usage

This work wa run across 3 platforms: 
* Local analysis on a desktop (Python notebooks)
* Image processing on a local supercomputer
* CNN training on Google's Cloud Platform (GCP)

This split isn't required just what was handy for our needs. Scripts have been tested on each platform and should be
able to run anywhere with just a change of directory values.

## Directory structure

The beginning of all notebooks and code contain constants for setting base directories to work for your individual 
setup. 

The local analysis setup is:
* BASE_DIR + /code/OAI 
* BASE_DIR + /code/xray_fingerprints

The image processing setup is:
* BASE_DIR +

The GCP structure is:
* BASE_DIR +

## Processing the OAI structured data

The OsteoArthritis Initiative (OAI) data set that this work is based on has countless uses. The CAIRO lab has a separate
repo created to contain code for pre-processing this data. Please see https://github.com/cairo-lab/pyOAI, and note where
to decompress the OAI data in the project README.md. The OAI data can be downloaded from https://nda.nih.gov/oai. 
Conversion of the SAS datafiles can be done by invoking the 
["Convert SAS to Dataframes"](https://github.com/cairo-lab/pyOAI/blob/main/notebooks/Convert%20SAS%20to%20Dataframes.ipynb) 
Python Jupyter notebook, this cleans and converts the original data into Pandas dataframes and stores them in Parquet
files (see the OAI code repo for full reasoning).

Input files:
* OAI/data/structured_data/OAICompleteData_SAS.zip

From this preprocessing you will create the following dataframes:
* allclinical_values.parquet
* enrollees_values.parquet

## Extracting the OAI image data

All OAI image data is several terabytes. As such, we do leave uncompressed versions of every image lying around. Instead
we initially scan all images and store metadata about the images in the compressed image archives. This metadata is then
used to select and only decompress the desired images.

Run OAI/notebooks/Get All Image Paths From Archives.ipynb to extract all image paths into xray_bilat_pa_fixed_flex_knee_values.parquet
. OAI structured data sometimes references images that aren't in the archives, so this serves as ground truth for what 
we have images for.

Some images are inverted (high pixel values are black instead of low ones). To know which files need their values
flipped we need to collect all DICOM metadata. Run OAI_DICOM/DICOM_Metadata_To_Pandas.ipynb to 
create dicom_metadata_df.parquet which has the metadata for each image.

Since all metadata is conveniently in a dataframe, we can simplify the raw image data into simple NumPy files (for
faster parsing later). This done through ....

Inputs files:
* Image archives: P01.zip. V00.zip, etc.

From this preprocessing you will create the following dataframes:
* xray_bilat_pa_fixed_flex_knee_values.parquet 
* dicom_metadata_df.parquet

## Cohort Creation

At this stage, all code is part of this repo (xray_fingerprints). However, it still uses a set of utility functions from
the CAIRO labs pyOAI code repo. On UNIX like platforms is important to create a soft-link to this library. In the notebook
directory (BASE_DIR + /code/xray_fingerprints/notebooks), run the following. 

`ln -s ../../OAI/notebooks/OAI_Utilities.py`

Use "Build Label Dataframe.ipynb" to then generate a dataframe containing image names and the associated deep learning
target variables. 

Input files:
* allclinical_values.parquet
* enrollees_values.parquet
* xray_bilat_pa_fixed_flex_knee_values.parquet 
* dicom_metadata_df.parquet

From this preprocessing you will create the following dataframes:
* data/xray_shortcutting_labels.parquet

## Preprocessing Knee Images

Use the script parallel_image_scaling.py to pre-process each image. This uses the cohort split from the prior step to
determine which images are training and which are test. The pre-processing can be set to either just z-score normalize 
images or to apply CLAHE first. All output images are 224x224.

Input files:
* data/xray_shortcutting_labels.parquet

From this preprocessing you will create the following dataframes:
* data/xray_shortcutting_labels.parquet

## Model Training and Testing

At this stage, both the images and modeling targets are ready. It is time for the model training.

## Table and Graph Creation

# Experiments





