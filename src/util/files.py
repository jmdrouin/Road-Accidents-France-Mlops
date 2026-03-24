from pathlib import Path

def single_file_in_folder(folder, pattern):
    folder = Path(folder)
    files = list(folder.glob(pattern))
    if len(files) != 1:
        print("[!] FILES FOUND:", files)
        raise RuntimeError(f"Expected exactly 1 file, found {len(files)}")
    return files[0]

def last_file_in_folder(folder, pattern):
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    if not files:
        raise RuntimeError("No files found")
    return files[-1]

