Write-Host "Setting up Python Environment..."
python --version
python -m venv venv

Write-Host "Installing Python Dependencies..."
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install numpy fastapi[standard] requests tqdm
.\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124

Write-Host "Cleaning previous builds..."
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "container_solver*.pyd") { Remove-Item "container_solver*.pyd" }

Write-Host "Building C++ Container Solver..."
mkdir build
cd build

# --- FIX: Force 64-bit Visual Studio Compiler ---
# Adjust "Visual Studio 17 2022" if you have an older version (e.g. 16 2019)
cmake ../../src -G "Visual Studio 17 2022" -A x64 -DPython_EXECUTABLE="../venv/Scripts/python.exe"
# ------------------------------------------------

cmake --build . --config Release

if (Test-Path "Release\container_solver*.pyd") {
    copy "Release\container_solver*.pyd" ..\
} elseif (Test-Path "container_solver*.pyd") {
    copy "container_solver*.pyd" ..\
}

cd ..
Write-Host "Setup Complete."