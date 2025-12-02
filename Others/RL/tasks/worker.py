import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/container-solver')))

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response, PlainTextResponse

from policy_value_network import PolicyValueNetwork
from package_utils import normalize_packages
from container_solver import Container

import numpy as np
import torch
import io

# Device setup (tries CUDA, then DirectML for AMD, then CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    try:
        import torch_directml
        device = torch_directml.device()
        print("Using DirectML device")
    except ImportError:
        pass

print(f"Worker running on: {device}")

policy_value_network = PolicyValueNetwork().to(device)
app = FastAPI()

@app.post('/policy_value_upload')
async def load_model(file: UploadFile = File(...)):
    content = await file.read()
    content = io.BytesIO(content)
    # weights_only=False is required for full model loading in this project structure
    policy_value_network.load_state_dict(torch.load(content, map_location=device, weights_only=False))
    policy_value_network.eval()
    return PlainTextResponse(content='success')

@app.post('/policy_value_inference')
async def root(request: Request):
    data = await request.body()
    
    try:
        batch_size = int(request.headers.get('batch-size', 1))
    except:
        batch_size = 1

    if batch_size > 1: 
        print(f'Inference for batch size {batch_size} received!')

    image_data = []
    packages_data = []
    
    if len(data) == 0:
        return Response(status_code=400, content="Empty data received")

    step_size = len(data) // batch_size
    
    # Process the binary data from C++
    containers = [Container.unserialize(data[i:i+step_size]) for i in range(0, len(data), step_size)]
    
    for container in containers:
        height_map = np.array(container.height_map, dtype=np.float32) / container.height
        image_data.append(np.expand_dims(height_map, axis=0))
        packages_data.append(normalize_packages(container))
    
    image_data = torch.tensor(np.stack(image_data, axis=0), device=device)
    packages_data = torch.tensor(np.stack(packages_data, axis=0), device=device)
    
    with torch.no_grad():
        policy, value = policy_value_network.forward(image_data, packages_data)
        policy = torch.softmax(policy, dim=1)
        result = torch.cat((policy, value), dim=1)

    result = result.cpu().numpy()
    return Response(content=result.tobytes(), media_type='application/octet-stream')