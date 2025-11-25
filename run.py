#!/usr/bin/env python3
import subprocess
import sys
import time
from pathlib import Path

def start_service(name, command, cwd=None, wait=3):
    print(f"Starting {name}...")
    
    try:
        if sys.platform == "win32":
            process = subprocess.Popen(
                command,
                cwd=cwd,
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        
        time.sleep(wait)
        print(f"{name} started")
        return process
        
    except Exception as e:
        print(f"Failed to start {name}: {e}")
        return None

def main():
    root_dir = Path(__file__).parent.resolve()
    processes = []
    
    # Start Alignment Service
    align_dir = root_dir / "align_image"
    if align_dir.exists():
        p1 = start_service("Alignment", "python server/app.py", cwd=str(align_dir), wait=5)
        processes.append(p1)
    
    # Start Classification Service
    classify_dir = root_dir / "classify_image"
    if classify_dir.exists():
        p2 = start_service("Classification", "python server/app.py", cwd=str(classify_dir), wait=5)
        processes.append(p2)
    
    # Start Gradio Web
    p3 = start_service("Web Interface", "python gra.py", cwd=str(root_dir), wait=5)
    processes.append(p3)
    
    print("\nAll services started!")
    print("Web: http://localhost:7878")
    print("\nPress Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        for p in processes:
            if p:
                try:
                    p.terminate()
                    p.wait(timeout=5)
                except:
                    p.kill()
        print("Done")

if __name__ == "__main__":
    main()