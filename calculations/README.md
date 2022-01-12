# Calclation.py Code

### calculateMS and calculateTorch
calculateMS function is used for calculating the mean and std for one image in np.ndarray that's in (height, width, channels) format
calculateTorch is used for calculating the mean and std for one image in torch tensor or np.ndarray that's in (channels, height, width) format

### calcMStd and calcMStdTorch
Basically the same as **calculateMS** and **calculateTorch** but calculates the mean and std for a whole dataset

### normalizeT and normalizeN
normalizes the dataset and returns a normalized dataset, uses **normalizeimgT** and **normalizeimgN** which normalizes one image only
