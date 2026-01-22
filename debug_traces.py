#!/usr/bin/env python3
"""
Debug script to check trace structure in pickle files
"""
import pickle
import sys
import glob

def check_traces(filepath):
    """Check trace structure in a pickle file"""
    print(f"\n{'='*60}")
    print(f"Checking: {filepath}")
    print(f"{'='*60}\n")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Check top-level keys
        print("Top-level keys:", list(data.keys())[:15])
        
        # Check traces
        all_traces = data.get('all_traces', [])
        all_voting_traces = data.get('all_voting_traces', [])
        
        print(f"\nall_traces length: {len(all_traces)}")
        print(f"all_voting_traces length: {len(all_voting_traces)}")
        
        # Check first trace structure
        if all_traces:
            print("\n=== First trace in all_traces ===")
            first_trace = all_traces[0]
            print("Keys:", list(first_trace.keys()))
            print("\nTrace content:")
            for key, value in first_trace.items():
                if key == 'text' or key == 'full_text':
                    print(f"  {key}: {str(value)[:200]}...")
                elif isinstance(value, (list, dict)):
                    print(f"  {key}: {type(value).__name__} (length: {len(value)})")
                else:
                    print(f"  {key}: {value}")
        
        if all_voting_traces:
            print("\n=== First trace in all_voting_traces ===")
            first_voting_trace = all_voting_traces[0]
            print("Keys:", list(first_voting_trace.keys()))
            print("\nTrace content:")
            for key, value in first_voting_trace.items():
                if key == 'text' or key == 'full_text':
                    print(f"  {key}: {str(value)[:200]}...")
                elif isinstance(value, (list, dict)):
                    print(f"  {key}: {type(value).__name__} (length: {len(value)})")
                else:
                    print(f"  {key}: {value}")
        
        # Check if text exists in any trace
        if all_traces:
            print("\n=== Checking for text field in all traces ===")
            text_found = 0
            full_text_found = 0
            for i, trace in enumerate(all_traces[:5]):  # Check first 5
                has_text = 'text' in trace and trace['text']
                has_full_text = 'full_text' in trace and trace['full_text']
                if has_text:
                    text_found += 1
                if has_full_text:
                    full_text_found += 1
                print(f"Trace {i}: text={has_text}, full_text={has_full_text}, keys={list(trace.keys())}")
            
            print(f"\nSummary: text found in {text_found}/{min(5, len(all_traces))} traces")
            print(f"         full_text found in {full_text_found}/{min(5, len(all_traces))} traces")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_traces.py <pkl_file>")
        print("Or: python debug_traces.py <pattern>")
        sys.exit(1)
    
    pattern = sys.argv[1]
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching: {pattern}")
        sys.exit(1)
    
    for filepath in files:
        check_traces(filepath)

