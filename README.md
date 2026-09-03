# Repository Structure

This repository contains two main folders.

## Geodesic Tracking

The first folder contains a collection of Jupyter notebooks that implement the segmentation of overlapping structures in SEM images using geodesic tracking.

### 1. Structure Filtering
Filters line and pillar structures from an image. The filter parameters are derived from synthetic examples.

### 2. Connected Component Algorithm
Identifies the connected components in the filtered image.

### 3. Morphological Edge Detection
Extracts the boundaries of the connected components. These boundaries serve as an initial estimate of the line and pillar contours and will be refined in later processing steps.

### 4. Position-Orientation Space Tessellation
Tessellates the position-orientation space by computing, for every voxel, the distance to the nearest connected-component contour.

### 5. Edge-Structure Filter
Detects and enhances the edges of structures present in the image.

### 6. Geodesic Tracking
Implements a fast B-spline approximation of geodesic tracking along the detected edges using the previously computed tessellation of position-orientation space.

### Pipeline Overview

The processing pipeline consists of the following steps:

1. Filter line and pillar structures from the input image.
2. Identify connected components.
3. Extract an initial estimate of component boundaries.
4. Tessellate the position-orientation space using distances to the extracted boundaries.
5. Detect structural edges.
6. Perform geodesic tracking using a fast B-spline approximation.

## U-Net Implementation

The second folder contains three Python files:

- `model.py` – Defines the U-Net architecture.
- `dataloader.py` – Loads and preprocesses the training data.
- `unet.py` – Trains the U-Net model.





