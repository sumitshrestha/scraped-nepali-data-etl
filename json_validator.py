#!/usr/bin/env python3
"""
Validate a large JSON file without loading the entire object into memory.
Requires the 'ijson' library for streaming validation.
"""

import sys
import json
import argparse

def validate_with_ijson(file_path):
    """Validate JSON using ijson (streaming, memory-efficient)."""
    try:
        import ijson
    except ImportError:
        raise ImportError(
            "ijson is required for streaming validation. "
            "Install it with: pip install ijson"
        )

    try:
        with open(file_path, 'rb') as f:
            # ijson.parse() returns a generator that yields events.
            # Just iterating through it validates the syntax.
            for _ in ijson.parse(f):
                pass
        return True
    except ijson.JSONError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return False

def validate_with_stdlib(file_path):
    """Validate JSON using standard json.load() (loads entire file)."""
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate a JSON file.")
    parser.add_argument('file', help='Path to the JSON file')
    parser.add_argument(
        '--legacy', action='store_true',
        help='Use standard json.load() (loads entire file, may use a lot of memory)'
    )
    args = parser.parse_args()

    if args.legacy:
        is_valid = validate_with_stdlib(args.file)
    else:
        try:
            is_valid = validate_with_ijson(args.file)
        except ImportError:
            print("ijson not found. Falling back to standard json.load().", file=sys.stderr)
            is_valid = validate_with_stdlib(args.file)

    sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main()