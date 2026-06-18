import os
import sys
import time
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

def download_file_chunked(url, outfile, expected_size=None):
    print(f"Downloading {os.path.basename(outfile)}...")
    os.makedirs(os.path.split(outfile)[0], exist_ok=True)
    
    # Try downloading with streaming and chunking
    chunk_size = 1024 * 1024 # 1 MB chunks
    downloaded = 0
    
    for attempt in range(5):
        try:
            headers = {}
            # If file already exists partially, we could support resume,
            # but since pre-signed URL might expire or change, let's start clean or check range support.
            # Let's just do a clean stream download first.
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            if expected_size:
                total_size = expected_size
                
            start_time = time.time()
            with open(outfile, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            speed = downloaded / (time.time() - start_time) / (1024 * 1024) # MB/s
                            sys.stdout.write(f"\rProgress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({percent:.1f}%) | Speed: {speed:.2f} MB/s")
                            sys.stdout.flush()
                        else:
                            sys.stdout.write(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB")
                            sys.stdout.flush()
            print("\nDownload completed successfully!")
            return True
        except Exception as e:
            print(f"\nAttempt {attempt + 1} failed: {e}")
            if attempt < 4:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("All download attempts failed.")
                return False

def main():
    print("Authenticating Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    
    username = api.config_values['username']
    kernel_slug = "fine-tune-qwen-coder-1-5b"
    kernel_id = f"{username}/{kernel_slug}"
    
    print(f"Listing output files for kernel {kernel_id}...")
    
    # Fetch all pages of output files
    all_files = []
    token = None
    
    with api.build_kaggle_client() as kaggle:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = username
            request.kernel_slug = kernel_slug
            if token:
                request.page_token = token
                
            response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
            all_files.extend(response.files)
            token = response.next_page_token
            if not token:
                break
                
    print(f"Found {len(all_files)} files in output.")
    
    # Filter for GGUF model and Modelfile
    target_files = []
    for item in all_files:
        filename = item.file_name
        if filename.endswith(".gguf") or filename == "Modelfile":
            target_files.append(item)
            
    if not target_files:
        print("No .gguf model or Modelfile found in output! Showing all files:")
        for item in all_files:
            print(f"- {item.file_name}")
        return
        
    for item in target_files:
        outfile = item.file_name # Download to current directory
        url = item.url
        # Download the file
        success = download_file_chunked(url, outfile)
        if not success:
            sys.exit(1)
            
    print("Finished downloading all target files!")

if __name__ == "__main__":
    main()
