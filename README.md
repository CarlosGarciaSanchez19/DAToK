# DAToK (Detector Acceptance Tool Kit)

This command-line tool allows you to compute and visualise DTs CMS detector acceptance maps (only barrel DTs). It supports different computation modes, output formats, and configurable verbosity levels.

---

## 📦 Installation

Clone or download the repository and ensure you have **Python 3.8+** installed.  
You can install the dependencies (if any) with:

```bash
cd Files
pip install -r requirements.txt
```

---

## 🚀 Usage

Run the program from the command line:

```bash
python DAToK.py [OPTIONS]
```

---

## ⚙️ Arguments

| Argument | Type | Description |
|-----------|------|-------------|
| `-v`, `--verbose` | flag (repeatable) | Sets the verbosity level. Without it, only saving messages are displayed. Add `-v` or `-vv` for more detailed output. |
| `-p`, `--plot` | flag | If included, generates and displays acceptance map plots. If not, only computes acceptances and saves them as `.npy` files. |
| `-s`, `--save` | flag | If included, computes and saves the maps as **C++ maps** in a C library format. If not, saves them as `.npy` files. |
| `--eta_method` | string | Specifies the method for computing η (eta) acceptance in (φ, η) maps. Valid options:<br> `SL2_0`, `SL2_L1`, `SL1_0`, `SL1_L2`.<br>Default: `SL2_L1`. |
| `--phi_method` | string | Specifies the method for computing φ (phi) acceptance in (φ, η) maps. Valid options:<br> `SL1_0`, `SL1_L1`.<br>Default: `SL1_L1`. |

---

## 🧩 Examples

### 1. Compute and save acceptances (default)
```bash
python DAToK.py
```
Computes acceptance maps and saves them as `.npy` files.

---

### 2. Compute and generate plots
```bash
python DAToK.py -p
```
Creates and displays acceptance map plots in addition to computing the data.

---

### 3. Save as C++ maps instead of `.npy`
```bash
python DAToK.py -s
```
Saves computed maps as C++ library files for use in C/C++ environments.

---

### 4. Increase verbosity
```bash
python DAToK.py -vv
```
Shows detailed logging of intermediate computations and results.

---

### 5. Specify computation methods
```bash
python DAToK.py --eta_method SL1_L2 --phi_method SL1_0
```
Uses custom methods for eta and phi acceptance computations.

---

### 6. Full example
```bash
python DAToK.py -v -p -s --eta_method SL2_0 --phi_method SL1_L1
```
Runs the tool in verbose mode, generates plots, and saves the results both as C++ maps and `.npy` files.

---

## 🧠 Notes

- If both `--plot` and `--save` are omitted, results are stored as `.npy` files by default.  
- Use multiple `-v` flags to increase verbosity (`-v`, `-vv`, etc.).  
- Default computation methods are:  
  - `eta_method = SL2_L1`  
  - `phi_method = SL1_L1`

---

## 📄 License

This project is distributed under the MIT License. See `LICENSE` for more details.
