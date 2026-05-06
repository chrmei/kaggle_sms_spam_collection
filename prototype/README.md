# Prototype Quickstart (Linux + Windows)

This folder has a minimal SMS spam pipeline:

- `train.py`: train and save model
- `evaluate.py`: evaluate model + test on real messages

---

## 1) Install `uv` (if not installed)

### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart terminal after install, then check:

```bash
uv --version
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart PowerShell, then check:

```powershell
uv --version
```

---

## 2) Create venv + install dependencies

Run these commands from the project root.

### Linux

```bash
uv venv .venv
source .venv/bin/activate
uv pip install pandas scikit-learn joblib jupyter
```

### Windows (PowerShell)

```powershell
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install pandas scikit-learn joblib jupyter
```

---

## 3) Train and evaluate

If your dataset is at `data/raw/spam.csv`:

### Linux

```bash
python prototype/train.py
python prototype/evaluate.py
```

### Windows (PowerShell)

```powershell
python prototype/train.py
python prototype/evaluate.py
```

If your CSV is somewhere else:

```bash
python prototype/train.py --data-path "path/to/spam.csv"
python prototype/evaluate.py --data-path "path/to/spam.csv"
```

---

## 4) Start Jupyter Notebook

### Linux

```bash
jupyter notebook
```

### Windows (PowerShell)

```powershell
jupyter notebook
```

Then open the shown local URL in your browser.

