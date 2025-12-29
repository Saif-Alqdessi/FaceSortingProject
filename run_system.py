"""
Master Orchestrator Script for AI Smart Event Photo Sorter

This script runs the complete pipeline sequentially:
1. Enrollment (enroll.py) - Generate face embeddings
2. Photo sorting (process_photos.py) - Classify and sort photos
3. Results distribution (send_results.py) - Send sorted photos via email
"""

import sys
import subprocess
import os
from pathlib import Path

# Script paths (UPDATE THESE if your script names are different)
ENROLL_SCRIPT = "src/enroll.py"           # Face enrollment script
SORTING_SCRIPT = "src/process_photos.py"  # Main photo sorting script
SEND_SCRIPT = "src/send_results.py"       # n8n automation script

BASE_DIR = Path(__file__).parent

def run_script(script_path, script_name):
    """
    Run a Python script using the current Python interpreter (venv-aware).
    
    Args:
        script_path: Path to the script file
        script_name: Display name for logging
        
    Returns:
        True if script executed successfully (exit code 0), False otherwise
    """
    script_full_path = BASE_DIR / script_path
    
    # Check if script exists
    if not script_full_path.exists():
        print(f"❌ [ERROR] Script not found: {script_full_path}")
        return False
    
    print(f"\n{'='*70}")
    print(f"🚀 [STARTING] {script_name}")
    print(f"📄 Script: {script_path}")
    print(f"🐍 Python: {sys.executable}")
    print(f"{'='*70}\n")
    
    try:
        # Use sys.executable to ensure we use the venv Python interpreter
        result = subprocess.run(
            [sys.executable, str(script_full_path)],
            cwd=str(BASE_DIR),
            check=False,  # Don't raise exception, we'll check return code
            stdout=sys.stdout,  # Stream output in real-time
            stderr=sys.stderr
        )
        
        if result.returncode == 0:
            print(f"\n{'='*70}")
            print(f"✅ [SUCCESS] {script_name} completed successfully!")
            print(f"{'='*70}\n")
            return True
        else:
            print(f"\n{'='*70}")
            print(f"❌ [FAILED] {script_name} exited with code {result.returncode}")
            print(f"{'='*70}\n")
            return False
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  [INTERRUPTED] {script_name} was interrupted by user")
        return False
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ [ERROR] Exception while running {script_name}: {e}")
        print(f"{'='*70}\n")
        return False

def main():
    """Main orchestrator function."""
    print("\n" + "="*70)
    print("🎯 AI SMART EVENT PHOTO SORTER - COMPLETE PIPELINE")
    print("="*70)
    print(f"📁 Working Directory: {BASE_DIR}")
    print(f"🐍 Python Interpreter: {sys.executable}")
    print("="*70)
    print("\n📋 Pipeline Steps:")
    print("   1️⃣  Enrollment (Generate face embeddings)")
    print("   2️⃣  Photo Sorting (Classify and organize photos)")
    print("   3️⃣  Results Distribution (Send via email)")
    print("="*70)
    
    # Step 1: Run enrollment script
    print("\n" + "="*70)
    print("📝 STEP 1: FACE ENROLLMENT")
    print("="*70)
    success = run_script(ENROLL_SCRIPT, "Face Enrollment")
    
    if not success:
        print("\n" + "="*70)
        print("🛑 [STOPPED] Pipeline stopped due to enrollment failure")
        print("   The embeddings.pkl file was not created.")
        print("   Photo sorting and distribution will NOT run.")
        print("="*70 + "\n")
        sys.exit(1)
    
    # Step 2: Run photo sorting script (only if enrollment succeeded)
    print("\n" + "="*70)
    print("📸 STEP 2: PHOTO SORTING")
    print("="*70)
    success = run_script(SORTING_SCRIPT, "Photo Sorting")
    
    if not success:
        print("\n" + "="*70)
        print("🛑 [STOPPED] Pipeline stopped due to sorting failure")
        print("   Photos were not sorted into person folders.")
        print("   Results distribution will NOT run.")
        print("="*70 + "\n")
        sys.exit(1)
    
    # Step 3: Run results distribution script (only if sorting succeeded)
    print("\n" + "="*70)
    print("📧 STEP 3: RESULTS DISTRIBUTION")
    print("="*70)
    success = run_script(SEND_SCRIPT, "Results Distribution")
    
    if not success:
        print("\n" + "="*70)
        print("⚠️  [WARNING] Distribution script failed, but sorting completed")
        print("   Photos are sorted in Data/output/ but emails were not sent.")
        print("="*70 + "\n")
        sys.exit(1)
    
    # All three steps completed successfully
    print("\n" + "="*70)
    print("🎉 [COMPLETE] Entire pipeline executed successfully!")
    print("="*70)
    print("✅ Step 1 - Face Enrollment: Complete")
    print("✅ Step 2 - Photo Sorting: Complete")
    print("✅ Step 3 - Results Distribution: Complete")
    print("="*70)
    print("\n✨ All event photos have been sorted and distributed!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  [INTERRUPTED] Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ [FATAL ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
