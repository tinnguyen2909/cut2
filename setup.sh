#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Set the base URL
BASE_URL="your_base_url_here"  # Replace with the actual base URL

# Function to download, extract and rename
download_extract_rename() {
    local filename=$1
    local target_name=$2
    local extract_opts=$3
    
    echo "Processing ${filename}..."
    wget "${BASE_URL}/${filename}" -O "${filename}"
    
    # Create a temporary directory for extraction
    local temp_dir="temp_${target_name}"
    mkdir -p "${temp_dir}"
    
    # Extract based on file type
    if [[ $extract_opts == "-J" ]]; then
        tar -xf "${filename}" -C "${temp_dir}"
    else
        tar -xzf "${filename}" -C "${temp_dir}"
    fi
    
    # Find the extracted directory and move to target
    local extracted_dir=$(find "${temp_dir}" -mindepth 1 -maxdepth 1 -type d | head -1)
    if [ -n "${extracted_dir}" ]; then
        mv "${extracted_dir}" "data/${target_name}"
    else
        # If no directory found, move all files
        mkdir -p "data/${target_name}"
        mv "${temp_dir}"/* "data/${target_name}/"
    fi
    
    # Clean up
    rm -rf "${temp_dir}" "${filename}"
    echo "Completed ${filename}"
}

# Launch all downloads and extractions in parallel
echo "Starting parallel downloads and extractions..."
download_extract_rename "A.tar.xz" "A" "-J" &
download_extract_rename "B.tar.gz" "B" "-z" &
download_extract_rename "flicker_faces_bg_removed_white_fill_toon_upscaled.tar.gz" "xB2" "-z" &
download_extract_rename "flickr_faces.tar.gz" "xA2" "-z" &

# Wait for all background processes to complete
wait
echo "All downloads and extractions completed"

# Clone GitHub repository
echo "Cloning GitHub repository..."
git clone https://github.com/tinnguyen2909/cut2.git

# Install dependencies
echo "Installing dependencies..."
cd cut2
pip install -r requirements.txt

echo "Setup complete!"