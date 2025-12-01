# 3D Bin Packing Solver

![FedEx Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/FedEx_Corporation_-_2016_Logo.svg/1920px-FedEx_Corporation_-_2016_Logo.svg.png)

## Main Solution: Strata

This solution utilizes a high-performance C++ implementation of the **Strata algorithm** to optimize 3D bin packing.

### 📂 File Structure

The core source code is located in the `Main/source` directory:

* `main.cpp`: The entry point containing the solver logic.
* `bin_pack.h` & `packing_structures.h`: Header files defining the algorithms and data structures.
* `CMakeLists.txt`: Build configuration file.
* `input.txt`: The input data file to be processed.

---

### ⚙️ Setup & Compilation

You can compile the project using **CMake**, **Visual Studio Code**, or manually using **g++**.

#### Option 1: Using Command Line (CMake)

1.  Ensure you have a C++ compiler and **CMake** installed.
2.  Open a terminal and navigate to the source directory:
    ```bash
    cd Main/source
    ```
3.  Create a build directory and compile the project:
    ```bash
    mkdir build
    cd build
    cmake ..
    cmake --build .
    ```

#### Option 2: Using VS Code

1.  Open the project folder in **VS Code**.
2.  Install the **C/C++** and **CMake Tools** extensions.
3.  Open the Command Palette (`Ctrl+Shift+P`) and select **CMake: Configure**.
4.  Press **F7** (or select **CMake: Build**) to compile the executable.

#### Option 3: Manual Compilation (g++)

If you prefer to compile manually without CMake, you can use `g++` directly. Ensure you use C++17 or newer (C++23 is also supported).

1.  Navigate to the source directory:
    ```bash
    cd Main/source
    ```
2.  Run the compile command:
    ```bash
    g++ -std=c++17 -O3 main.cpp -o main
    ```
    *(Note: You can replace `-std=c++17` with `-std=c++23` if your compiler supports it.)*

---

### 🚀 How to Run

1.  **Prepare Input:**
    Ensure your `input.txt` file is formatted correctly according to the problem statement and placed in the same directory as the executable.

2.  **Execute:**
    Run the compiled executable from the terminal.

    **Windows (PowerShell):**
    ```powershell
    # If compiled via CMake:
    .\build\Debug\main.exe
    
    # If compiled manually via g++:
    .\main.exe
    ```

    **Linux/macOS:**
    ```bash
    # If compiled via CMake:
    ./build/main

    # If compiled manually via g++:
    ./main
    ```

3.  **Check Output:**
    The program will generate an `output.txt` file in the current working directory containing the final packing configuration.

---

## 🛠️ Validation & Visualization

### Validator

To verify that the output satisfies all constraints (no overlaps, weight limits, etc.):

1.  Navigate to the `Other Approaches/validator` folder.
2.  Run the validation script using Python:
    ```bash
    python validate.py --input ../../Main/source/input.txt --output ../../Main/source/output.txt
    ```

### Visualizer

To interactively view the 3D packing result:

1.  Open VS Code and navigate to `Other Approaches/visualizer`.
2.  Ensure you have the **"Live Server"** extension installed.
3.  Right-click `index.html` and select **"Open with Live Server"**.
4.  In the browser window that opens:
    * **Step 1:** Upload `input.txt` (or `feed.json`) to load package metadata.
    * **Step 2:** Upload the generated `output.txt`.