#!/bin/bash

# Simple script to run generate_isles_images.py in background
nice -n 10 ionice -c 2 -n 7 python generate_isles_images.py > generate_isles_output.log 2>&1 &
echo "Generation started in background. Check generate_isles_output.log for progress."
echo "Process ID: $!" 