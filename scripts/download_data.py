"""
Download full dataset from Kaggle using kagglehub.
This script downloads the complete road accidents dataset from Kaggle.
"""

import shutil
import traceback
from pathlib import Path

def download_dataset():
    """Download the full dataset from Kaggle."""
    print("=" * 70)
    print("DOWNLOADING FULL DATASET FROM KAGGLE")
    print("=" * 70)
    
    try:
        # Import kagglehub
        print("\nImporting kagglehub...")
        import kagglehub
        
        # Download latest version
        print("\nDownloading dataset (this may take a few minutes)...")
        print("   Dataset: ahmedlahlou/accidents-in-france-from-2005-to-2016")
        
        path = kagglehub.dataset_download("ahmedlahlou/accidents-in-france-from-2005-to-2016")
        
        print("\nDownload complete!")
        print(f"Dataset location: {path}")
        
        # List downloaded files
        print("\nDownloaded files:")
        dataset_path = Path(path)
        if dataset_path.exists():
            files = list(dataset_path.glob("*.csv"))
            for file in sorted(files):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   - {file.name} ({size_mb:.1f} MB)")

        data_root = "./data/csv"                
        print(f"\nCopying files to {data_root}")

        required_files = [
            'caracteristics.csv',
            'places.csv', 
            'users.csv',
            'vehicles.csv',
            'holidays.csv'
        ]
            
        copied_count = 0
        for filename in required_files:
            source = dataset_path / filename
            dest = Path(data_root, filename)
            
            if source.exists():
                try:
                    shutil.copy2(source, dest)
                    print(f"   Copied {filename}")
                    copied_count += 1
                except Exception as e:
                    print(f"   [X] Error copying {filename}: {e}")
            else:
                print(f"   [!] {filename} not found in download")
        
        print(f"\nCopied {copied_count}/{len(required_files)} files")
        
        print("\n" + "=" * 70)
        print("DATASET READY!")
        print("=" * 70)
        print("\nNext steps:")
        print("   1. Run the Streamlit app: streamlit run streamlit-app.py")
        print("   2. Or run the Jupyter notebooks in order:")
        print("      - Step_1_Data mining_DataViz.ipynb")
        print("      - Step 2_Pre-processing_feature-eng.ipynb")
        print("      - Step-3_Modeling_FINAL.ipynb")
        print("      - Bin_Modeling.ipynb")
        
        return True
        
    except ImportError:
        print("\n[!] Error: kagglehub not installed")
        print("\nInstall it with:")
        print("   pip install kagglehub")
        return False
        
    except Exception:
        print(f"\n[!] Error downloading dataset:")
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("   1. Ensure you have Kaggle API credentials configured")
        print("   2. Visit: https://www.kaggle.com/settings/account")
        print("   3. Create and download API token (kaggle.json)")
        print("   4. Place it in: ~/.kaggle/kaggle.json (Unix) or C:\\Users\\<user>\\.kaggle\\kaggle.json (Windows)")
        return False

def main():
    """Main function."""
    print("\nRoad Accidents in France - Data Download Utility")
    download_dataset()

if __name__ == "__main__":
    main()
