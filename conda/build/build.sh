#!/bin/bash
echo "Building Conda environment 'genai_project'..."
conda env create -f conda/build/requirements.yaml
echo "Build complete."