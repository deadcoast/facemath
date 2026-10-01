#!/bin/sh
set -eu
out=${1:-runs/wave10}
python -m facemath milestone-lab --out "$out"
python -m facemath overview "$out/positive/qualification"
python -m facemath overview "$out/negative/qualification"
python -m facemath overview "$out/admitted_pair"
