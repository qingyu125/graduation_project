#!/usr/bin/env python3
"""
Simple script to view pickle file contents
"""
import pickle
import json
import sys
import argparse
from pprint import pprint

def view_pkl(filepath: str, pretty: bool = True, keys_only: bool = False, key: str = None):
    """View contents of a pickle file"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        if keys_only:
            if isinstance(data, dict):
                print("Top-level keys:")
                for k in data.keys():
                    print(f"  - {k}")
            else:
                print(f"Data type: {type(data)}")
                if hasattr(data, '__dict__'):
                    print("Attributes:")
                    for k in data.__dict__.keys():
                        print(f"  - {k}")
            return
        
        if key:
            if isinstance(data, dict) and key in data:
                print(f"\n=== Value for key '{key}' ===")
                if pretty:
                    pprint(data[key], width=120, depth=10)
                else:
                    print(data[key])
            else:
                print(f"Key '{key}' not found in data")
                if isinstance(data, dict):
                    print(f"Available keys: {list(data.keys())[:20]}")
            return
        
        print(f"\n=== Pickle File: {filepath} ===\n")
        print(f"Data type: {type(data)}\n")
        
        if isinstance(data, dict):
            print("Top-level keys:")
            for k in sorted(data.keys()):
                value = data[k]
                if isinstance(value, (list, dict)):
                    print(f"  - {k}: {type(value).__name__} (length: {len(value)})")
                else:
                    print(f"  - {k}: {type(value).__name__} = {str(value)[:100]}")
            
            if pretty:
                print("\n=== Full Content ===")
                pprint(data, width=120, depth=5)
        else:
            if pretty:
                pprint(data, width=120, depth=10)
            else:
                print(data)
                
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='View pickle file contents')
    parser.add_argument('filepath', type=str, help='Path to pickle file')
    parser.add_argument('--keys-only', action='store_true', 
                       help='Show only top-level keys')
    parser.add_argument('--key', type=str, 
                       help='Show value for specific key')
    parser.add_argument('--no-pretty', action='store_true',
                       help='Disable pretty printing')
    
    args = parser.parse_args()
    
    view_pkl(args.filepath, pretty=not args.no_pretty, 
             keys_only=args.keys_only, key=args.key)

if __name__ == "__main__":
    main()

