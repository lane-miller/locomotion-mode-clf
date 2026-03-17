# Data

## Dataset

**PAMAP2 Physical Activity Monitoring** — wearable IMU data from 9 subjects 
performing a range of physical activities, collected at Universität Passau.

- 9 subjects, 18 activities (subset used here)
- IMU sensors: wrist, chest, ankle (accelerometer, gyroscope, magnetometer)
- IMU sampling rate: 100Hz
- Format: plain-text CSV, pandas-compatible

## Source

UCI Machine Learning Repo:
https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring

## Local Storage

Data is stored on external SSD, not committed to this repo

Local path: `<SSD>/Datasets/Bio/PAMAP2/`

## Note on Camargo et al.

Originally intended to use the Camargo et al. (2021) lower-limb 
biomechanics dataset, which stores sensor data as MATLAB table objects 
incompatible with Python-based loading libraries. PAMAP2 is compatible alternative

## Citation

Reiss, A. (2012). PAMAP2 Physical Activity Monitoring [Dataset]. 
UCI Machine Learning Repository. 
https://doi.org/10.24432/C5NW2H