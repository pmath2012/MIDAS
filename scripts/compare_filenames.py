import os
import re
import argparse

def extract_id_slice(fname):
    m = re.match(r'(test_sub-stroke\d+_\d+)', fname)
    return m.group(1) if m else None

def get_id_set(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    return set(filter(None, (extract_id_slice(f) for f in files)))

def unmatched_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    unmatched = [f for f in files if not re.match(r'(test_sub-stroke\d+_\d+)', f)]
    return unmatched

def main(dir1, dir2):
    ids1 = get_id_set(dir1)
    ids2 = get_id_set(dir2)
    only_in_1 = ids1 - ids2
    only_in_2 = ids2 - ids1
    print(f"IDs only in {dir1} ({len(only_in_1)}):")
    for i in sorted(only_in_1):
        print(i)
    print(f"\nIDs only in {dir2} ({len(only_in_2)}):")
    for i in sorted(only_in_2):
        print(i)
    print(f"\nIDs in both: {len(ids1 & ids2)}")
    # Print unmatched files
    unmatched1 = unmatched_files(dir1)
    unmatched2 = unmatched_files(dir2)
    print(f"\nUnmatched PNG files in {dir1} ({len(unmatched1)}):")
    for f in unmatched1:
        print(f)
    print(f"\nUnmatched PNG files in {dir2} ({len(unmatched2)}):")
    for f in unmatched2:
        print(f)

def rename_files(directory):
    # Get all files in the directory
    files = os.listdir(directory)
    
    # Filter for png files with the specific suffix
    target_files = [f for f in files if f.endswith('_ncct_fake_B.png')]
    
    print(f"Found {len(target_files)} files to rename")
    
    # Rename each file
    for old_name in target_files:
        # Create new name by replacing the suffix
        new_name = old_name.replace('_ncct_fake_B.png', '_dwi.png')
        old_path = os.path.join(directory, old_name)
        new_path = os.path.join(directory, new_name)
        
        # Check if destination file already exists
        if os.path.exists(new_path):
            print(f"Warning: {new_name} already exists, skipping {old_name}")
            continue
            
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {old_name} -> {new_name}")
        except Exception as e:
            print(f"Error renaming {old_name}: {str(e)}")

def compare_files(dir1, dir2):
    # ... existing comparison code ...
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare or rename files.")
    subparsers = parser.add_subparsers(dest='command')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare')
    compare_parser.add_argument('dir1', help='First directory')
    compare_parser.add_argument('dir2', help='Second directory')
    
    # Rename command
    rename_parser = subparsers.add_parser('rename')
    rename_parser.add_argument('--dir', default='scripts/isles_gen/resvit', 
                             help='Directory containing files to rename')
    
    args = parser.parse_args()
    
    if args.command == 'compare':
        compare_files(args.dir1, args.dir2)
    elif args.command == 'rename':
        rename_files(args.dir)
    else:
        parser.print_help() 