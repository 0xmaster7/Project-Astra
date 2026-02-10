# Image Processing Module

## Overview
This module provides functionality for loading and normalizing images for processing by the neural network.

## Key Components

### `load_and_normalize_image(directory_path, extensions=['.jpg', '.jpeg', '.png', '.bmp'])`
- **Purpose**: Loads all images with specified extensions from a directory and normalizes them
- **Parameters**:
  - `directory_path`: Path to the directory containing images
  - `extensions`: List of valid file extensions to process
- **Returns**: Normalized image data as numpy arrays

### `normalize_image(image_path)`
- **Purpose**: Opens and normalizes one or more images
- **Parameters**:
  - `image_path`: Single path string or list of path strings to images
- **Process**:
  - Converts images to RGB if needed
  - Converts image data to numpy arrays
  - Normalizes pixel values to range [0.0, 1.0]
- **Returns**: Normalized image data as numpy arrays

## Usage
This module is used as the first step in the image processing pipeline, preparing raw images for input to the UNet neural network.