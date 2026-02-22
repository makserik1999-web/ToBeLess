# RWF-2000 Dataset Setup

## Download Instructions

The RWF-2000 dataset requires signing an agreement to access. Follow these steps:

1. **Visit the official repository**: https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection

2. **Sign the agreement**: Download and fill out the agreement form from the repository

3. **Request access**: Send the signed form to get the download link

4. **Extract to this folder** with the following structure:

```
datasets/
└── RWF-2000/
    ├── train/
    │   ├── Fight/
    │   │   ├── video001.avi
    │   │   ├── video002.avi
    │   │   └── ...
    │   └── NonFight/
    │       ├── video001.avi
    │       ├── video002.avi
    │       └── ...
    └── val/
        ├── Fight/
        │   └── ...
        └── NonFight/
            └── ...
```

## Dataset Statistics

- **Total videos**: 2,000 clips
- **Classes**: 2 (Fight, NonFight)
- **Split**: ~1,600 train, ~400 val (80/20)
- **Source**: Real-world CCTV surveillance footage

## Alternative: Kaggle Mirror

Some users have uploaded mirrors to Kaggle. Search for "RWF-2000" on Kaggle.

## Verification

Run `python verify_dataset.py` to check the dataset structure after extraction.
