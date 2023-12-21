#!/bin/bash

mkdir models
wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth -O ./models/codeformer.pth
wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth -O ./models/RealESRGAN_x2plus.pth
wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth -O ./models/parsing_parsenet.pth
wget https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth -O ./models/detection_Resnet50_Final.pth