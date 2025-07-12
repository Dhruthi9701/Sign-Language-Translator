import numpy as np
import os

def examine_keypoints_structure():
    """Examine the structure of keypoints data"""
    base_path = "ISL_Keypoints"
    
    print("=== ISL_Keypoints Folder Structure Analysis ===\n")
    
    # Check main folder
    if not os.path.exists(base_path):
        print(f"Error: {base_path} folder not found!")
        return
    
    # Examine each subfolder
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            print(f"\n--- Folder: {folder} ---")
            
            # Check subfolders
            subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            if subfolders:
                print(f"Subfolders: {subfolders}")
                
                for subfolder in subfolders:
                    subfolder_path = os.path.join(folder_path, subfolder)
                    npy_files = [f for f in os.listdir(subfolder_path) if f.endswith('.npy')]
                    
                    print(f"\n  Subfolder: {subfolder}")
                    print(f"  Number of .npy files: {len(npy_files)}")
                    
                    if npy_files:
                        # Examine first file
                        first_file = os.path.join(subfolder_path, npy_files[0])
                        try:
                            data = np.load(first_file)
                            print(f"  First file: {npy_files[0]}")
                            print(f"  Data shape: {data.shape}")
                            print(f"  Data type: {data.dtype}")
                            print(f"  File size: {os.path.getsize(first_file)} bytes")
                            
                            # Show sample data
                            if len(data.shape) == 2:
                                print(f"  Sample data (first 3 frames, first 10 features):")
                                print(data[:3, :10])
                            else:
                                print(f"  Sample data (first 10 values):")
                                print(data.flatten()[:10])
                                
                        except Exception as e:
                            print(f"  Error reading {first_file}: {e}")
            else:
                # Direct .npy files in folder
                npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
                print(f"Direct .npy files: {len(npy_files)}")
                if npy_files:
                    print(f"Sample files: {npy_files[:5]}")

if __name__ == "__main__":
    examine_keypoints_structure() 