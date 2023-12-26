#!/bin/bash

mkdir -p models

download_file() {
    url=$1
    output=$2
    if [ ! -f "$output" ]; then
        wget "$url" -O "$output"
    else
        echo "$output already exists. Skipping download."
    fi
}

download_file "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth" "./models/codeformer.pth"
download_file "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth" "./models/RealESRGAN_x2plus.pth"
download_file "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth" "./models/parsing_parsenet.pth"
download_file "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth" "./models/detection_Resnet50_Final.pth"