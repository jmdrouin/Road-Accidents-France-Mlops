"""
Create sample datasets from full CSV files for demonstrative purposes.
Samples 10,000 rows from each dataset while maintaining data distribution.
"""

import pandas as pd
import os
from pathlib import Path

def create_sample_data(input_file, output_file, n_rows=10000, random_state=42):
    """
    Create a sample dataset from a full CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output sample CSV file
        n_rows: Number of rows to sample (default: 10000)
        random_state: Random seed for reproducibility (default: 42)
    """
    try:
        print(f"\n📂 Processing {input_file}...")
        
        # Determine encoding
        encoding = 'latin-1' if 'caracteristics' in input_file or 'users' in input_file or 'vehicles' in input_file else 'utf-8'
        
        # Read full dataset
        print(f"   Loading data (encoding: {encoding})...")
        df = pd.read_csv(input_file, encoding=encoding, low_memory=False)
        total_rows = len(df)
        print(f"   Total rows: {total_rows:,}")
        
        # Sample data
        if total_rows > n_rows:
            print(f"   Sampling {n_rows:,} rows...")
            df_sample = df.sample(n=n_rows, random_state=random_state)
        else:
            print(f"   File has fewer than {n_rows:,} rows, keeping all...")
            df_sample = df.copy()
        
        # Save sample
        print(f"   Saving to {output_file}...")
        df_sample.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"   ✅ Created sample with {len(df_sample):,} rows")
        
        return True
        
    except FileNotFoundError:
        print(f"   ❌ Error: {input_file} not found")
        return False
    except Exception as e:
        print(f"   ❌ Error processing {input_file}: {str(e)}")
        return False

def main():
    """Main function to create all sample datasets."""
    print("=" * 60)
    print("🔬 Creating Sample Datasets for GitHub Repository")
    print("=" * 60)
    
    # Create sample_data directory
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    print(f"\n📁 Sample directory: {sample_dir.absolute()}")
    
    # Define file mappings (input -> output)
    files_to_sample = {
        'caracteristics.csv': 'sample_data/caracteristics_sample.csv',
        'places.csv': 'sample_data/places_sample.csv',
        'users.csv': 'sample_data/users_sample.csv',
        'vehicles.csv': 'sample_data/vehicles_sample.csv',
    }
    
    # Files to copy as-is (small files)
    files_to_copy = ['holidays.csv']
    
    # Create samples
    print("\n" + "=" * 60)
    print("📊 CREATING SAMPLES (10,000 rows each)")
    print("=" * 60)
    
    success_count = 0
    for input_file, output_file in files_to_sample.items():
        if create_sample_data(input_file, output_file, n_rows=10000):
            success_count += 1
    
    # Copy small files
    print("\n" + "=" * 60)
    print("📋 COPYING SMALL FILES")
    print("=" * 60)
    
    for file in files_to_copy:
        try:
            print(f"\n📂 Copying {file}...")
            if os.path.exists(file):
                import shutil
                shutil.copy(file, f"sample_data/{file}")
                print(f"   ✅ Copied {file}")
                success_count += 1
            else:
                print(f"   ⚠️ {file} not found, skipping")
        except Exception as e:
            print(f"   ❌ Error copying {file}: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully processed: {success_count}/{len(files_to_sample) + len(files_to_copy)} files")
    
    # List created files
    print(f"\n📁 Sample files created in {sample_dir.absolute()}:")
    if sample_dir.exists():
        for file in sorted(sample_dir.iterdir()):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   - {file.name} ({size_mb:.2f} MB)")
    
    print("\n" + "=" * 60)
    print("✅ Sample dataset creation complete!")
    print("=" * 60)
    print("\n💡 To use sample data in Streamlit app:")
    print("   1. Modify streamlit-app.py to load from sample_data/ folder")
    print("   2. Or move sample files to root directory")
    print("\n💡 To recreate samples with different size:")
    print("   python create_samples.py --rows 5000")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create sample datasets for GitHub repository')
    parser.add_argument('--rows', type=int, default=10000, help='Number of rows to sample (default: 10000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Override if command line args provided
    if args.rows != 10000:
        print(f"\n⚙️ Using custom sample size: {args.rows:,} rows")
        # Re-run with custom size (simple approach)
        for input_file in ['caracteristics.csv', 'places.csv', 'users.csv', 'vehicles.csv']:
            output_file = f"sample_data/{input_file.replace('.csv', '_sample.csv')}"
            create_sample_data(input_file, output_file, n_rows=args.rows, random_state=args.seed)
    else:
        main()
