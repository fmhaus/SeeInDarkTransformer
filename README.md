# SeeInDarkTransformer

SeeInTheDarkTransformer is a transformer-based adaptation of "Learning to See in the Dark" in CVPR 2018, by Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun.
[Paper](https://cchen156.github.io/paper/18CVPR_SID.pdf)
[Github Implementation](https://github.com/cchen156/Learning-to-See-in-the-Dark)

## Setup

- Download the dataset using `download_dataset_sony.sh` or [Link](https://storage.googleapis.com/isl-datasets/SID/Sony.zip)
- Used libraries: 
    - rawpy: reading raw image filesd
    - numpy: general data processing
    - torch: machine learning
    - boto3: S3 storage for data and checkpoints
    - tkinter: Demo and image comparison
    - pillow: Image loading
    - torchprofile: profiling