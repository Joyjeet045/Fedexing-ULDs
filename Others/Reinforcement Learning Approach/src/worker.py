from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import Response, PlainTextResponse

from policy_value_network import PolicyValueNetwork
from package_utils import normalize_packages
from container_solver import Container

import numpy as np
import torch

import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device != 'cuda':
  try:
    import torch_directml
    device = torch_directml.device()
  except ImportError:
    pass
  
policy_value_network = PolicyValueNetwork().to(device)
app = FastAPI()

@app.post('/policy_value_upload')
async def load_model(file: UploadFile = File(...)):
  content = await file.read()
  content = io.BytesIO(content)
  policy_value_network.load_state_dict(torch.load(content, weights_only=False))
  policy_value_network.eval()
  return PlainTextResponse(content='success')

@app.post('/policy_value_inference')
async def root(request: Request):
  data = await request.body()
  
  batch_size = int(request.headers['batch-size'])
  if batch_size > 1: print(f'Inference for batch size {batch_size} received!')

  image_data = []
  packages_data = []
  step_size = len(data) // batch_size
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