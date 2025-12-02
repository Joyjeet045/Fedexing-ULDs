# Container Packing with MCTS & Deep Learning

RL system solving 3D bin packing using Monte Carlo Tree Search + neural networks.

## Quick Start

```bash
# Terminal 1: Worker runs automatically (port 8000)

# Terminal 2: Train
cd RL/tasks
python train.py --iteration_count 1 --worker_addresses 127.0.0.1:8000

# Terminal 3: Inference
python inference_final.py --worker 127.0.0.1:8000 --output results.txt
```

**Output Format:**
```
3276.0,13,0
P-0,NONE,-1,-1,-1,-1,-1,-1
P-22,U3,0.0,0.0,0.0,6.0,7.0,6.0
```
Line 1: `cost,packages_packed,priority_boxes` | Rest: `P-ID,BOX,x1,y1,z1,x2,y2,z2`

---

## System Architecture

### Information Flow

```
┌──────────────────────────────────────────────────────────┐
│                    Training Loop (Python)                │
└──────────────────────────────────────────────────────────┘
                            ↓
    ┌───────────────────────────────────────────────┐
    │         Episode Generation (C++)              │
    │  • MCTS explores 64 simulations/move          │
    │  • Generates training states & targets        │
    └───────────────┬─────────────────────────────┘
                    │ HTTP POST /predict
                    ↓
    ┌──────────────────────────────────────────────┐
    │    FastAPI Worker (Python, port 8000)        │
    │  • Loads trained model (PyTorch)             │
    │  • Returns policy & value predictions        │
    └──────────────────────────────────────────────┘
                    ↑ HTTP Response
                    │
    ┌───────────────┴──────────────────────────────┐
    │                                              │
    │     PyTorch Neural Network                   │
    │     (GPU: CUDA/DirectML/CPU)                 │
    │                                              │
    └──────────────────────────────────────────────┘
                    ↓
    ┌──────────────────────────────────────────────┐
    │      Data Augmentation & Batch Training      │
    │  • Rotations (90°, 180°, 270°)               │
    │  • Reflections (horizontal/vertical)         │
    │  • SGD with policy + value loss              │
    └──────────────────────────────────────────────┘
```

### Neural Network Architecture

```
Input Layer
    ├─ Container Height Map (16×16)     [2D CNN]
    └─ Unplaced Package Info (32×4)     [Features]
           ↓
    ┌──────────────────────────┐
    │   ResidualTower Block    │
    │  (Conv + ReLU + Residual)│
    │  ×3 layers              │
    └──────────────────────────┘
           ↓
    ┌──────────────┬──────────────┐
    │              │              │
    ↓              ↓              ↓
Policy Head    Value Head    Shared Features
  (256 units)   (1 unit)     (512 units)
     ↓              ↓
Action Probs    Scalar Value
```

**Design**: AlphaZero-style dual-head network for both action selection (policy) and state evaluation (value).

### Container State Representation

```
Container Model:
  Dimensions: 16×16×24 (L×W×H)
  Capacity: 1 container (U3)
  
Height Map (2D Top-Down View):
  ┌─────────────────────┐
  │ 5 3 0 0 2 0 0 0 ... │  ← Height at each (x, y)
  │ 4 2 0 1 3 0 0 0 ... │
  │ 0 0 0 0 0 0 0 0 ... │
  │ ...                 │
  └─────────────────────┘
  
Package Placement: At lowest available position (greedy height-based)

Packages: 32 random, dimensions 4-8 units each
```

### Data Flow: Training

```
[Iteration i]
    │
    ├─→ MCTS Episode (8 games)
    │   ├─ Game 1: [State₀ → Action → State₁ → ... → Terminal]
    │   ├─ Game 2: [State₀ → Action → State₁ → ... → Terminal]
    │   └─ ...
    │
    └─→ Generate Training Data
        ├─ π_targets (from MCTS visit counts)
        ├─ v_targets (from final rewards)
        └─ s_states (container height maps + packages)
            │
            ├─→ Data Augmentation (8× via rotations/reflections)
            │
            └─→ Batch Training (PyTorch)
                ├─ Forward pass
                ├─ Loss = α·PolicyLoss + β·ValueLoss
                ├─ Backprop
                └─ Update weights
```

---

## Directory Structure

```
RL/
├── README.md
├── CMakeLists.txt              (Build config)
├── requirements.txt            (Python deps)
│
├── src/                        (C++ engine)
│   ├── container_solver.cc     (pybind11 bindings)
│   ├── container.h/cc          (State management)
│   ├── package.h               (Data structure)
│   ├── mcts.h                  (Tree search)
│   └── array2d.h               (Height map)
│
├── build/                      (CMake artifacts)
│
└── tasks/                      (Python training & inference)
    ├── train.py                (Main training loop)
    ├── worker.py               (FastAPI inference server)
    ├── policy_value_network.py (PyTorch model)
    ├── package_utils.py        (Utilities)
    ├── inference_final.py      (Run inference)
    └── policy_value_network.pth (Weights)
```

---

## How It Works

### Training Pipeline
1. **MCTS Episode**: C++ generates games using tree search guided by neural network
2. **Data Augmentation**: Python applies rotations/reflections to increase dataset
3. **Batch Training**: PyTorch trains on augmented data with policy + value loss
4. **Weight Update**: Improved model saved and deployed to worker

### Inference Pipeline
1. **MCTS Search**: Explores 64 simulations per move using current model
2. **Worker Prediction**: HTTP calls to FastAPI for policy/value guidance
3. **Package Placement**: Selects action with highest policy confidence
4. **Output**: CSV with 3D coordinates of each placed package

---

## Configuration

### Training Hyperparameters (in `train.py`)
```python
iterations           = 1         # Training loops
simulations_per_move = 64        # MCTS depth
episodes_per_iter    = 8         # Games per iteration
batch_size          = 16        # Training batch
learning_rate       = 0.001     # SGD rate
```

### Hardware Requirements
- **GPU**: NVIDIA CUDA (recommended), AMD DirectML, or CPU fallback
- **Memory**: ~4GB for model + training data
- **Time**: ~8 minutes per iteration (CPU), ~2 minutes (GPU)

---

## Dependencies

| Layer | Component | Purpose |
|-------|-----------|---------|
| **C++** | GLM, CPR, pybind11 | Math, HTTP, Python bridge |
| **Python** | PyTorch, NumPy | Deep learning, numerics |
| **Web** | FastAPI, Requests | Inference server, HTTP |
| **Build** | CMake, GCC | C++20 compilation |

---

## Rebuild C++ Module

```bash
cd RL/build
cmake .. -DCPR_USE_SYSTEM_CURL=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON
make -j4
# Output: RL/src/container-solver/container_solver*.so
```

---

## Design Rationale

| Design Decision | Reason |
|---|---|
| **Hybrid C++/Python** | C++ for speed (MCTS), Python for ML flexibility |
| **Height Map State** | 2D CNN-friendly, memory-efficient vs full 3D voxels |
| **HTTP Workers** | Scales inference independently from search |
| **Dual-Head Network** | Separate policy (action) and value (evaluation) predictions |
| **MCTS** | Self-play learning without labeled data |

---

## Quick Troubleshooting

| Issue | Fix |
|---|---|
| Worker not responding | `curl http://127.0.0.1:8000/health` |
| CUDA not found | `python -c "import torch; print(torch.cuda.is_available())"` |
| C++ import error | Rebuild: `cd RL/build && cmake .. && make -j4` |
| Training slow | Use GPU or reduce `simulations_per_move` |

---

**Status**: Production-ready | **Updated**: December 2025
