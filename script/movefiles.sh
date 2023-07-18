#!/bin/bash

SRC_DIR=/scratch1/09498/janechen/mydata/tragen-output-offline-1x
DST_DIR=/scratch2/09498/janechen

# Check if the source directory exists
if [ ! -d "$SRC_DIR" ]; then
  echo "Source directory not found: $SRC_DIR"
  exit 1
fi

# Check if the destination directory exists
if [ ! -d "$DST_DIR" ]; then
  echo "Destination directory not found: $DST_DIR"
  exit 1
fi

# Function to check if a directory is non-empty
function is_directory_non_empty() {
  dir="$1"
  if [ -z "$(ls -A "$dir")" ]; then
    return 1 # Directory is empty
  else
    return 0 # Directory is non-empty
  fi
}

# Iterate through each directory in DST_DIR
for dir in "$DST_DIR"/*; do
  if [ -d "$dir" ]; then
    if is_directory_non_empty "$dir"; then
      # Copy the non-empty directory from SRC_DIR to DST_DIR using rsync
      rsync -av "$SRC_DIR/$(basename "$dir")/" "$dir/"
    fi
  fi
done

echo "rsync of non-empty directories completed."