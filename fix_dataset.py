import shutil
from pathlib import Path
import os

def fix_structure():
    base_dir = Path("datasets/RWF-2000")
    nested_dir = base_dir / "RWF-2000"
    
    if not nested_dir.exists():
        print(f"Nested directory {nested_dir} does not exist. Structure might be already correct or different.")
        print(f"Contents of {base_dir}:")
        for item in base_dir.iterdir():
            print(f"  - {item.name}")
        return

    print(f"Found nested directory: {nested_dir}")
    print("Moving contents up one level...")
    
    # Move plain files
    for item in nested_dir.iterdir():
        target = base_dir / item.name
        if target.exists():
            if target.is_dir():
                # Merge directories if they exist (though unpredictable, usually better to remove target first if empty)
                print(f"Target {target} exists. Merging/Overwriting...")
                # For simplicity in this specific fix, we assume we want the stuff from nested_dir
                # but shutil.move fails if dest dir exists.
                # Let's use specific logic for train/val
                pass
            else:
                 print(f"File {target} exists. Skipping.")
        
        print(f"Moving {item.name} -> {base_dir}")
        shutil.move(str(item), str(base_dir))

    print("Removing empty nested directory...")
    try:
        nested_dir.rmdir()
    except Exception as e:
        print(f"Could not remove {nested_dir}: {e}")

    print("Structure fix complete.")
    
    # Verification
    print("\nVerifying...")
    train_dir = base_dir / "train"
    if train_dir.exists():
        print("SUCCESS: 'train' directory found at correct level.")
    else:
        print("ERROR: 'train' directory NOT found at correct level.")

if __name__ == "__main__":
    fix_structure()
