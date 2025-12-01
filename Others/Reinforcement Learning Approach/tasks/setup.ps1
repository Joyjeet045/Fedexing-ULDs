python --version
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy fastapi[standard] requests tqdm
pip install torch --index-url https://download.pytorch.org/whl/cu124