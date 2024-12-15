from pathlib import Path

# Get the base directory of the project (where this script resides)
BASE_DIR = Path.cwd()

# Define the index directory in one line
INDEX_DIR = BASE_DIR / "var" / "concert_index"

# Ensure INDEX_DIR is a string if needed elsewhere
INDEX_DIR = str(INDEX_DIR)