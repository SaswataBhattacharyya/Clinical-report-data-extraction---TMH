import torch

def check_gpu():
    """Check CUDA/GPU availability and return status"""
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else None
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    
    return {
        "available": cuda_available,
        "version": cuda_version,
        "device_name": device_name
    }

if __name__ == "__main__":
    result = check_gpu()
    print("CUDA Available:", result["available"])
    print("CUDA Version:", result["version"])
    print("Device:", result["device_name"])
