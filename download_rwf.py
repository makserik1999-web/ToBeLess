import os
import shutil
import sys
import subprocess
from pathlib import Path

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import kagglehub
except ImportError:
    print("Installing kagglehub...")
    install_package("kagglehub")
    import kagglehub

def setup_dataset():
    print("Downloading RWF-2000 dataset from Kaggle...")
    # Download latest version
    path = kagglehub.dataset_download("vulamnguyen/rwf2000")
    
    print(f"Dataset downloaded to cache: {path}")
    
    # Target directory
    target_dir = Path("datasets/RWF-2000")
    
    # Create target directory
    if target_dir.exists():
        print(f"Target directory {target_dir} already exists. Cleaning up...")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Moving files to {target_dir}...")
    
    # The dataset structure might vary, let's explore the downloaded path
    source_path = Path(path)
    
    # Iterate over files in source and move them
    for item in source_path.iterdir():
        if item.is_dir():
            shutil.copytree(item, target_dir / item.name)
        else:
            shutil.copy2(item, target_dir / item.name)
            
    print("\nDataset setup complete!")
    print(f"Location: {target_dir.absolute()}")
    
    # Verify structure
    train_dir = target_dir / "train"
    val_dir = target_dir / "val"
    
    if train_dir.exists() and val_dir.exists():
        print("\nStructure verification: SUCCESS [train/ and val/ found]")
    else:
        print("\nStructure verification: WARNING [Expected train/ and val/ folders]")
        print(f"Contents of {target_dir}:")
        for item in target_dir.iterdir():
            print(f"  - {item.name}")

if __name__ == "__main__":
    setup_dataset()
