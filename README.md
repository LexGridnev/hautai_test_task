# Description
The task is to segment image pixels by the color on the image in the 
`image` folder.

# Requirements

You need python >= 3.8. Other requirements are in `requirements.txt`.

# How to run
Please do the following:
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3.8 python3.8-dev python3.8-venv
git clone this repo
cd into this repo
python3.8 -m venv hautai_test_task
source hautai_test_task/bin/activate
pip install -r requirements.txt
```
and run `demo.ipynb` to see clustering results with different parameters.

# Key features

1. KMeans pixel clustering;
2. Cartesian and polar (Angle only for provided image) pixel coordinates as 
   additional features;
3. Ability to choose image channels for clustering algorithm.

# Structure

* `segmentation.py` - contains core clustering algorithm;
* `demo.ipynb` - contains clustering results with different parameters.
