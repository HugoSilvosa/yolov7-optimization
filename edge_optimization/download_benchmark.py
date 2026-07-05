import os
import urllib.request
import stat
from pathlib import Path

def main():
    url = "https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_x86-64_benchmark_model"
    output_path = Path("bin/benchmark_model")
    
    print(f"Downloading benchmark_model from {url}...")
    try:
        # Download the precompiled binary
        urllib.request.urlretrieve(url, output_path)
        print("Download completed successfully.")
        
        # Make the binary executable
        print("Setting executable permissions (chmod +x)...")
        st = os.stat(output_path)
        os.chmod(output_path, st.st_mode | stat.S_IEXEC)
        print("Permissions set successfully.")
        
        # Verify file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Verified: benchmark_model is {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"ERROR downloading benchmark_model: {e}")
        if output_path.exists():
            output_path.unlink()

if __name__ == "__main__":
    main()
