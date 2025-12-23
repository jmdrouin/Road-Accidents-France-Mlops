"""
Cleanup script to remove generated files and artifacts before publishing to GitHub.
This script helps prepare the repository for a clean commit.
"""

import os
import shutil
from pathlib import Path
import argparse

def remove_file(filepath, verbose=True):
    """Remove a file if it exists."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            if verbose:
                print(f"   ✅ Removed: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"   ❌ Error removing {filepath}: {e}")
        return False

def remove_directory(dirpath, verbose=True):
    """Remove a directory and its contents if it exists."""
    try:
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)
            if verbose:
                print(f"   ✅ Removed directory: {dirpath}")
            return True
        return False
    except Exception as e:
        print(f"   ❌ Error removing {dirpath}: {e}")
        return False

def cleanup_generated_data(dry_run=False):
    """Remove generated data files (keep originals)."""
    print("\n📊 GENERATED DATA FILES")
    print("-" * 60)
    
    files_to_remove = [
        'acc.csv',              # Merged dataset
        'master_acc.csv',       # ML-ready dataset
    ]
    
    removed = 0
    for file in files_to_remove:
        if os.path.exists(file):
            if not dry_run:
                if remove_file(file):
                    removed += 1
            else:
                print(f"   [DRY RUN] Would remove: {file}")
                removed += 1
    
    if removed == 0:
        print("   ℹ️ No generated data files found")
    
    return removed

def cleanup_python_cache(dry_run=False):
    """Remove Python cache files and directories in project only."""
    print("\n🐍 PYTHON CACHE FILES")
    print("-" * 60)
    
    removed = 0
    # Exclude common environment and dependency directories
    exclude_patterns = ['.conda', 'venv', 'env', 'node_modules', '.git']
    
    # Helper function to check if path should be excluded
    def should_exclude(path_str):
        path_parts = Path(path_str).parts
        # Check if any part matches or starts with exclude patterns
        return any(
            any(part == pat or part.startswith('.' + pat) or part.startswith(pat + '-') 
                for pat in exclude_patterns)
            for part in path_parts
        )
    
    # Remove __pycache__ directories
    for pycache in Path('.').rglob('__pycache__'):
        if should_exclude(str(pycache)):
            continue
        if not dry_run:
            if remove_directory(str(pycache)):
                removed += 1
        else:
            print(f"   [DRY RUN] Would remove: {pycache}")
            removed += 1
    
    # Remove .pyc files
    for pyc in Path('.').rglob('*.pyc'):
        if should_exclude(str(pyc)):
            continue
        if not dry_run:
            if remove_file(str(pyc)):
                removed += 1
        else:
            print(f"   [DRY RUN] Would remove: {pyc}")
            removed += 1
    
    # Remove .pyo files
    for pyo in Path('.').rglob('*.pyo'):
        if should_exclude(str(pyo)):
            continue
        if not dry_run:
            if remove_file(str(pyo)):
                removed += 1
        else:
            print(f"   [DRY RUN] Would remove: {pyo}")
            removed += 1
    
    if removed == 0:
        print("   ℹ️ No Python cache files found")
    
    return removed

def cleanup_jupyter_checkpoints(dry_run=False):
    """Remove Jupyter notebook checkpoints."""
    print("\n📓 JUPYTER CHECKPOINTS")
    print("-" * 60)
    
    removed = 0
    for checkpoint in Path('.').rglob('.ipynb_checkpoints'):
        if not dry_run:
            if remove_directory(str(checkpoint)):
                removed += 1
        else:
            print(f"   [DRY RUN] Would remove: {checkpoint}")
            removed += 1
    
    if removed == 0:
        print("   ℹ️ No Jupyter checkpoints found")
    
    return removed

def cleanup_temp_scripts(dry_run=False):
    """Remove temporary scripts and test files."""
    print("\n🗑️ TEMPORARY SCRIPTS")
    print("-" * 60)
    
    files_to_remove = [
        'cleanup.py',           # Old cleanup script
        'export_models.py',     # Export script (functionality now in notebook)
        'replace_pages.py',     # Development utility
        'tmp_geo_check.py',     # Temporary test script
        'test.ipynb',           # Test notebook
        'Modeling.txt',         # Text notes
    ]
    
    removed = 0
    for file in files_to_remove:
        if os.path.exists(file):
            if not dry_run:
                if remove_file(file):
                    removed += 1
            else:
                print(f"   [DRY RUN] Would remove: {file}")
                removed += 1
    
    if removed == 0:
        print("   ℹ️ No temporary scripts found")
    
    return removed

def cleanup_old_documentation(dry_run=False):
    """Remove old documentation files."""
    print("\n📄 OLD DOCUMENTATION")
    print("-" * 60)
    
    files_to_remove = [
        'QUICK_START.md',           # Replaced by comprehensive README
        'RESTRUCTURING_SUMMARY.md', # Development notes
    ]
    
    removed = 0
    for file in files_to_remove:
        if os.path.exists(file):
            if not dry_run:
                if remove_file(file):
                    removed += 1
            else:
                print(f"   [DRY RUN] Would remove: {file}")
                removed += 1
    
    if removed == 0:
        print("   ℹ️ No old documentation found")
    
    return removed

def cleanup_virtual_envs(dry_run=False):
    """Remove virtual environment directories."""
    print("\n🔧 VIRTUAL ENVIRONMENTS")
    print("-" * 60)
    
    venv_dirs = ['.venv-py312', 'venv', '.venv', '.conda']
    
    removed = 0
    for venv in venv_dirs:
        if os.path.exists(venv):
            size_mb = sum(f.stat().st_size for f in Path(venv).rglob('*') if f.is_file()) / (1024 * 1024)
            if not dry_run:
                confirm = input(f"   ⚠️ Remove {venv}/ ({size_mb:.1f} MB)? (y/n): ").strip().lower()
                if confirm == 'y':
                    if remove_directory(venv):
                        removed += 1
            else:
                print(f"   [DRY RUN] Would remove: {venv}/ ({size_mb:.1f} MB)")
    
    if removed == 0:
        print("   ℹ️ No virtual environments removed")
    
    return removed

def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(description='Clean up repository before GitHub publication')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without actually removing')
    parser.add_argument('--all', action='store_true', help='Clean everything (data, cache, temp files, docs)')
    parser.add_argument('--data', action='store_true', help='Clean generated data files only')
    parser.add_argument('--cache', action='store_true', help='Clean Python cache only')
    parser.add_argument('--jupyter', action='store_true', help='Clean Jupyter checkpoints only')
    parser.add_argument('--temp', action='store_true', help='Clean temporary scripts only')
    parser.add_argument('--docs', action='store_true', help='Clean old documentation only')
    parser.add_argument('--venv', action='store_true', help='Remove virtual environments')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🧹 REPOSITORY CLEANUP UTILITY")
    print("=" * 70)
    
    if args.dry_run:
        print("\n⚠️ DRY RUN MODE - No files will be actually removed")
    
    total_removed = 0
    
    # If --all is specified or no specific flags, clean everything
    if args.all or not any([args.data, args.cache, args.jupyter, args.temp, args.docs, args.venv]):
        total_removed += cleanup_generated_data(args.dry_run)
        total_removed += cleanup_python_cache(args.dry_run)
        total_removed += cleanup_jupyter_checkpoints(args.dry_run)
        total_removed += cleanup_temp_scripts(args.dry_run)
        total_removed += cleanup_old_documentation(args.dry_run)
        if args.all:  # Only clean venv if explicitly requested with --all
            total_removed += cleanup_virtual_envs(args.dry_run)
    else:
        # Clean only specified categories
        if args.data:
            total_removed += cleanup_generated_data(args.dry_run)
        if args.cache:
            total_removed += cleanup_python_cache(args.dry_run)
        if args.jupyter:
            total_removed += cleanup_jupyter_checkpoints(args.dry_run)
        if args.temp:
            total_removed += cleanup_temp_scripts(args.dry_run)
        if args.docs:
            total_removed += cleanup_old_documentation(args.dry_run)
        if args.venv:
            total_removed += cleanup_virtual_envs(args.dry_run)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    if args.dry_run:
        print(f"Would remove: {total_removed} items")
        print("\n💡 Run without --dry-run to actually remove files")
    else:
        print(f"✅ Removed: {total_removed} items")
    
    print("\n" + "=" * 70)
    print("✅ CLEANUP COMPLETE!")
    print("=" * 70)
    
    if not args.dry_run and total_removed > 0:
        print("\n📋 Files kept:")
        print("   ✓ Jupyter notebooks (.ipynb)")
        print("   ✓ Source data files (caracteristics.csv, places.csv, etc.)")
        print("   ✓ Sample data (sample_data/ folder)")
        print("   ✓ Trained models (models/ folder)")
        print("   ✓ Streamlit app (streamlit-app.py)")
        print("   ✓ Documentation (README.md, MODEL_INTEGRATION.md)")
        print("   ✓ Utility scripts (download_data.py, create_samples.py)")
        
        print("\n🚀 Repository is now clean and ready for GitHub!")
        print("\n💡 Next steps:")
        print("   1. Review changes: git status")
        print("   2. Stage files: git add .")
        print("   3. Commit: git commit -m 'Clean repository for publication'")
        print("   4. Push to GitHub: git push origin main")

if __name__ == "__main__":
    main()
