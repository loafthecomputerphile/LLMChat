Below is a **generated README.md file** you can drop directly into your repo.
It is written as a **section** of a larger README, exactly as requested.

---

# 📦 Binaries Setup

This section documents the subsystem responsible for downloading, extracting, installing, and preparing external binaries (Pandoc, Ollama, and their associated models) used by the project.
It consists of two main files:

* `binaries_setup.py` — the executable installer logic
* `binaries_setup.toml` — configuration for URLs, scripts, and model settings

---

## 🔍 What This Module Does

This module automates the full lifecycle of preparing all external dependencies required by the larger system. Specifically, it:

### **1. Downloads Release Assets from GitHub**

* Fetches release metadata for *Pandoc* and *Ollama*
* Filters assets based on allowed archive types (`.zip`, `.tgz`, `.tar.gz`)
* Selects the correct binary for the target OS + architecture

### **2. Extracts and Installs Binaries**

* Safely extracts archives into `${ROOT}/data/bin`
* Automatically strips the top-level wrapper folder when present
* Automatically backs up prior installs as `*.backup`
* Restores the previous version if extraction fails

### **3. Generates Platform-Specific Portable Launchers**

The TOML file contains templates for fully portable launch scripts:

* **Windows:** creates `ollama_portable.bat`
* **macOS/Linux:** creates `ollama_portable.sh`

These scripts configure:

* `OLLAMA_MODELS`
* `OLLAMA_HOME`
* `OLLAMA_HOST`
* and invoke the embedded `ollama` binary inside the installation directory

### **4. Pulls and Installs Models**

If the installer is run on the *current* operating system, it will:

* mark the portable Ollama script as executable (Unix)
* pull the embedding model (`nomic-embed-text`)
* pull the LLM model (e.g., `Qwen3-ABL-1.7b`)
* rename/copy/remove the temp model files according to the script logic

### **5. Fully Portable Deployment**

Everything is placed under:

```
/data/
    ├── bin/
    │   ├── pandoc/
    │   ├── ollama/
    │   │     └── ollama_portable(.bat|.sh)
    │   └── ...
    │
    └── ollama_data/
        ├── models/
        └── ollama_home/
```

The system intentionally avoids modifying the system-level PATH or user-wide application directories.

---

## 🛠 How to Use It

### **1. Ensure both files exist**

```
binaries_setup.py
binaries_setup.toml
```

Both must reside in the same directory.

### **2. Run the installer**

#### **Windows**

```bash
python binaries_setup.py --os windows
```

#### **Linux**

```bash
python3 binaries_setup.py --os linux
```

#### **macOS**

```bash
python3 binaries_setup.py --os darwin
```

### **3. Optional: specify versions**

If you want a specific release instead of `latest`, supply flags:

```
--pandoc-version 3.2
--ollama-version v0.3.12
```

Example:

```bash
python binaries_setup.py --os windows --pandoc-version 3.2
```

### **4. After the install**

You will have a portable `ollama_portable` script inside:

```
data/bin/ollama/
```

Use it like:

```
ollama_portable serve
ollama_portable pull llama3
ollama_portable run mymodel
```

---

## 📁 Resulting File Structure (After Running the Installer)

The final layout is inferred from the Python file and the TOML scripts.

```
project_root/
│
├── binaries_setup.py
├── binaries_setup.toml
│
└── data/
    ├── bin/
    │   ├── pandoc/
    │   │   ├── <extracted pandoc binaries>
    │   │   └── ...
    │   │
    │   └── ollama/
    │       ├── bin/                   # Linux/macOS: where extracted 'ollama' lives
    │       ├── ollama.exe             # Windows version (if applicable)
    │       ├── ollama_portable.bat    # Windows launcher
    │       ├── ollama_portable.sh     # Unix launcher
    │       └── ...
    │
    └── ollama_data/
        ├── models/                    # downloaded models
        │   ├── nomic-embed-text
        │   ├── Qwen3-ABL-1.7b
        │   └── ...
        │
        └── ollama_home/               # Ollama runtime directory

```

### Notes on Structure

* `data/bin/**` always holds **binaries only**
* `ollama_data/**` is always created **one directory above the script** from the launcher logic
* Temporary backup installs appear as e.g.:

```
data/bin/pandoc.backup
data/bin/ollama.backup
```

---

