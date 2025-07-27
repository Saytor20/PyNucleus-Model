#!/bin/bash

# This script reorganizes the project structure according to the improvement plan.

# Create new directories
mkdir -p data/raw
mkdir -p data/processed/vector_store
mkdir -p data/models
mkdir -p data/outputs
mkdir -p data/validation_reports

# Move contents
mv data/01_raw/* data/raw/
mv data/02_processed/* data/processed/
mv data/03_intermediate/vector_db/* data/processed/vector_store/
mv data/03_processed/chromadb/* data/processed/vector_store/
mv data/04_models/* data/models/
mv data/05_output/* data/outputs/
mv data/validation/* data/validation_reports/

# Remove old directories
rm -rf data/01_raw
rm -rf data/02_processed
rm -rf data/03_intermediate
rm -rf data/03_processed
rm -rf data/04_models
rm -rf data/05_output
rm -rf data/validation

echo "Project structure has been reorganized."
