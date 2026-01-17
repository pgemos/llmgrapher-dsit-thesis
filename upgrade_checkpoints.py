# ==============================================================================
# CHECKPOINT UPGRADE SCRIPT
# ==============================================================================
# This script updates the keys within existing checkpoint JSON files to match
# the new naming convention introduced in 'llmgrapher_experiment_withPairs-new5.py'.
#
# It performs the following renames within each JSON file:
# - 'default_llm'       -> 'llm_vectordb_cr'
# - 'user_llm'          -> 'llm_inherent_cr'
#
# USAGE:
# 1. Place this script in the same directory as your 'checkpoints' folder.
# 2. Run the script using: python upgrade_checkpoints.py
# ==============================================================================

import os
import json
from pathlib import Path

# --- Configuration ---
CHECKPOINT_DIR = "checkpoints"
OLD_KEYS_MAP = {
    "default_llm": "llm_vectordb_cr",
    "user_llm": "llm_inherent_cr"
}

def upgrade_checkpoint_files():
    """
    Scans the checkpoint directory and updates the keys in any JSON files
    that use the old naming convention.
    """
    checkpoint_path = Path(CHECKPOINT_DIR)
    if not checkpoint_path.is_dir():
        print(f"Error: Checkpoint directory '{CHECKPOINT_DIR}' not found.")
        print("Please make sure you are running this script from the correct location.")
        return

    print(f"Scanning for checkpoints to upgrade in '{CHECKPOINT_DIR}'...")
    
    # Get a list of all JSON files in the directory
    json_files = list(checkpoint_path.glob('*.json'))
    
    if not json_files:
        print("No checkpoint files (.json) found to process.")
        return

    updated_count = 0
    skipped_count = 0

    for filepath in json_files:
        print(f"\nProcessing '{filepath.name}'...")
        made_changes = False
        
        try:
            # Read the entire JSON file into memory
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check for and rename each of the old keys
            for old_key, new_key in OLD_KEYS_MAP.items():
                if old_key in data:
                    print(f"  Found old key '{old_key}'. Renaming to '{new_key}'.")
                    # Copy the data to the new key and remove the old one
                    data[new_key] = data.pop(old_key)
                    made_changes = True
            
            if made_changes:
                # If any changes were made, write the updated data back to the file
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
                print(f"  Successfully updated and saved '{filepath.name}'.")
                updated_count += 1
            else:
                print("  File is already up-to-date. No changes needed.")
                skipped_count += 1

        except json.JSONDecodeError:
            print(f"  Warning: Could not decode JSON from '{filepath.name}'. Skipping file.")
            skipped_count += 1
        except Exception as e:
            print(f"  An unexpected error occurred while processing '{filepath.name}': {e}")
            skipped_count += 1

    print("\n" + "="*30 + " UPGRADE COMPLETE " + "="*30)
    print(f"Summary:")
    print(f"  - Files successfully updated: {updated_count}")
    print(f"  - Files skipped (already up-to-date or errors): {skipped_count}")
    print(f"  - Total files processed: {len(json_files)}")
    print("="*68)


if __name__ == "__main__":
    upgrade_checkpoint_files()