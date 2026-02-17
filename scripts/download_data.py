"""
Download full dataset from Kaggle using kagglehub.
This script downloads the complete road accidents dataset from Kaggle.
"""

import shutil
from pathlib import Path

def download_dataset():
    """Download the full dataset from Kaggle."""
    print("=" * 70)
    print("📥 DOWNLOADING FULL DATASET FROM KAGGLE")
    print("=" * 70)
    
    try:
        # Import kagglehub
        print("\n📦 Importing kagglehub...")
        import kagglehub
        
        # Download latest version
        print("\n⬇️ Downloading dataset (this may take a few minutes)...")
        print("   Dataset: ahmedlahlou/accidents-in-france-from-2005-to-2016")
        
        path = kagglehub.dataset_download("ahmedlahlou/accidents-in-france-from-2005-to-2016")
        
        print(f"\n✅ Download complete!")
        print(f"📁 Dataset location: {path}")
        
        # List downloaded files
        print(f"\n📋 Downloaded files:")
        dataset_path = Path(path)
        if dataset_path.exists():
            files = list(dataset_path.glob("*.csv"))
            for file in sorted(files):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   - {file.name} ({size_mb:.1f} MB)")
        
        # Optionally copy files to project root
        print("\n" + "=" * 70)
        copy_choice = input("📂 Copy files to project root directory? (y/n): ").strip().lower()
        
        if copy_choice == 'y':
            print("\n📋 Copying files to project root...")
            
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
                dest = Path(filename)
                
                if source.exists():
                    try:
                        shutil.copy2(source, dest)
                        print(f"   ✅ Copied {filename}")
                        copied_count += 1
                    except Exception as e:
                        print(f"   ❌ Error copying {filename}: {e}")
                else:
                    print(f"   ⚠️ {filename} not found in download")
            
            print(f"\n✅ Copied {copied_count}/{len(required_files)} files")
        
        print("\n" + "=" * 70)
        print("✅ DATASET READY!")
        print("=" * 70)
        print("\n🚀 Next steps:")
        print("   1. Run the Streamlit app: streamlit run streamlit-app.py")
        print("   2. Or run the Jupyter notebooks in order:")
        print("      - Step_1_Data mining_DataViz.ipynb")
        print("      - Step 2_Pre-processing_feature-eng.ipynb")
        print("      - Step-3_Modeling_FINAL.ipynb")
        print("      - Bin_Modeling.ipynb")
        
        return True
        
    except ImportError:
        print("\n❌ Error: kagglehub not installed")
        print("\n📦 Install it with:")
        print("   pip install kagglehub")
        return False
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {str(e)}")
        print("\n💡 Troubleshooting:")
        print("   1. Ensure you have Kaggle API credentials configured")
        print("   2. Visit: https://www.kaggle.com/settings/account")
        print("   3. Create and download API token (kaggle.json)")
        print("   4. Place it in: ~/.kaggle/kaggle.json (Unix) or C:\\Users\\<user>\\.kaggle\\kaggle.json (Windows)")
        return False

def main():
    """Main function."""
    print("\n🚗 Road Accidents in France - Data Download Utility")
    download_dataset()

if __name__ == "__main__":
    main()
