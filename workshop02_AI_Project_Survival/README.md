# Real-ESRGAN: enlarge one image with Python 3.13

## Project

**Repository:** [Real-ESRGAN on GitHub](https://github.com/xinntao/Real-ESRGAN)

**Inference task:** Enlarge the example image `0030.jpg` from **220 × 220** to **880 × 880** pixels using an already-trained model. “Inference” means running that model on an image; no training is needed.

## Repo map

| Item | What we use |
| --- | --- |
| Environment | Python **3.13.14**, uv, and a new `.venv` folder containing this project's Python packages. Tested on Linux using the CPU. |
| Entry point | `inference_realesrgan.py` — the Python file that runs the model. |
| Model | `RealESRGAN_x4plus`; its downloaded model file is `weights/RealESRGAN_x4plus.pth`. |
| Input | `inputs/0030.jpg` — included with the project. |
| Output | `results/0030_out.jpg` — the enlarged image. |

## Environment setup

These instructions are for a **Linux terminal using bash or zsh**, such as the workshop computers. Git, curl and uv must already be installed. Check with:

```bash
git --version
curl --version
uv --version
```

If any command says “command not found”, install that tool using your computer's setup instructions before continuing.

**Copy each complete command block, paste it into the terminal, and press Enter. Run the steps in order in the same terminal.** Wait until the prompt returns before continuing. If a command reports an error, stop at that step.

### 1. Download a fresh copy and select Python 3.13

**$\color{#FC0FC0}{\text{This folder called "realesrgan\_practice" is created in your HOME directory}}$**

This creates a new working folder OUTSIDE your `MLMPworkshop` submission repository. 

```bash
mkdir -p ~/realesrgan_practice
cd ~/realesrgan_practice
project_dir="Real-ESRGAN_fresh_$(date +%Y%m%d_%H%M%S)"
git clone https://github.com/xinntao/Real-ESRGAN.git "$project_dir"
cd "$project_dir"
git checkout a4abfb2979a7bbff3f69f58f58ae324608821e27
uv python install 3.13.14
uv venv --python 3.13.14 .venv
source .venv/bin/activate
python --version
which python
```

You should see **Python 3.13.14** and a Python path ending in **`.venv/bin/python`**. This is a new environment; do not activate your class `.venv` or `.venv313`.

Stay in this folder for every remaining step.

### 2. Install the supporting packages

The expandable block below contains the package versions recorded from the successful environment. You do not need to understand or edit the list. **Expand it and paste the entire block.** Downloads may take several minutes.

`--no-deps` installs this complete list without automatically installing BasicSR yet. We fix and install BasicSR in Step 3 because its original setup code can fail on Python 3.13.

<details>
<summary><strong>EXPAND ME EXPAND ME EXPAND ME to show the package-installation block</strong></summary>

```bash
cat > reproduction-packages.txt <<'PINS'
ImageIO==2.38.0
Jinja2==3.1.6
Markdown==3.11
MarkupSafe==3.0.4
PyYAML==6.0.3
Werkzeug==3.1.9
absl-py==2.5.0
addict==2.4.0
certifi==2026.7.22
charset-normalizer==3.5.2
contourpy==1.4.0
cuda-bindings==13.4.3
cuda-pathfinder==1.8.3
cuda-toolkit==13.0.3.0
cycler==0.12.1
facexlib==0.3.0
filelock==4.0.12
filterpy==1.4.5
fonttools==4.66.1
fsspec==2026.9.0
future==1.0.0
gfpgan==1.3.8
grpcio==1.84.0
idna==3.20
kiwisolver==1.5.1
lazy-loader==0.6
llvmlite==0.50.0
lmdb==3.0.0
matplotlib==3.11.2
mpmath==1.3.0
networkx==3.7
numba==0.68.0
numpy==2.5.3
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.24.0.43
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu13==2.30.7
nvidia-nvjitlink==13.4.92
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
opencv-python==5.0.0.93
packaging==26.3
pillow==12.3.0
platformdirs==4.12.4
protobuf==7.36.2
pyparsing==3.3.3
python-dateutil==2.9.0.post0
requests==2.34.2
scikit-image==0.26.0
scipy==1.18.1
setuptools==84.0.0
six==1.17.0
sympy==1.14.0
tb-nightly==2.21.0a20251023
tensorboard-data-server==0.7.2
tifffile==2026.9.20
torch==2.14.1
torchvision==0.29.1
tqdm==4.70.1
triton==3.8.0
typing_extensions==4.16.0
urllib3==2.8.0
yapf==0.43.0
PINS
uv pip install --python .venv/bin/python --no-deps -r reproduction-packages.txt
```

</details>

**$\color{#FC0FC0}{YES \ Paste\ the\ whole\ thing}$**

### 3. Fix the Python 3.13 compatibility problems

**The important change is in `setup.py`: replace the entire `get_version()` function.** It is a Python function, not a terminal command or a comment. The original version can fail with `KeyError: '__version__'` under Python 3.13.

The block below makes that edit automatically. It applies the same fix to BasicSR, a library used by Real-ESRGAN, and fixes BasicSR's outdated torchvision import.

First download BasicSR's source code:

```bash
curl -fL https://files.pythonhosted.org/packages/86/41/00a6b000f222f0fa4c6d9e1d6dcc9811a374cabb8abb9d408b77de39648c/basicsr-1.4.2.tar.gz -o basicsr-1.4.2.tar.gz
```

Now **paste this whole block**, starting with `python - <<'PY'` and ending with the final `PY`. Those first and last lines allow Python code to run from your terminal. Do not paste just the `def get_version()` lines into zsh.

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import tarfile

archive = Path('basicsr-1.4.2.tar.gz')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == 'b89b595a87ef964cda9913b4d99380ddb6554c965577c0c10cb7b78e31301e87'
with tarfile.open(archive) as package:
    package.extractall('vendor', filter='data')

replacement = """def get_version():
    namespace = {}
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'), namespace)
    return namespace['__version__']
"""

for filename in ('setup.py', 'vendor/basicsr-1.4.2/setup.py'):
    path = Path(filename)
    source = path.read_text()
    start = source.index('def get_version():')
    end = source.index('\n\ndef ', start + 1)
    path.write_text(source[:start] + replacement + source[end:])

path = Path('vendor/basicsr-1.4.2/basicsr/data/degradations.py')
source = path.read_text().replace(
    'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
    'from torchvision.transforms.functional import rgb_to_grayscale')
path.write_text(source)
print('Compatibility fixes applied.')
PY
```

Expect `Compatibility fixes applied.` Then install the corrected library and the local project:

```bash
uv pip install --python .venv/bin/python --no-deps ./vendor/basicsr-1.4.2
uv pip install --python .venv/bin/python --no-deps -e .
```

The last command also creates `realesrgan/version.py`. Installing only `requirements.txt` does not complete this project setup.

### 4. Download the trained model

A “checkpoint” is the file containing what the model learned during training. Download it into `weights`:

```bash
mkdir -p weights
curl -fL https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -o weights/RealESRGAN_x4plus.pth
```

## Inference

### 5. Enlarge one image

Copy and run:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/0030.jpg -o results --outscale 4 --fp32
```

This selects the model, reads `0030.jpg`, enlarges it 4× and saves the result in `results`. The first two settings limit CPU threads; `--fp32` uses full precision. A GPU is not required.

You should see:

```text
Testing 0 0030
```

**Wait for the terminal prompt to return.** The message means processing started; the next step checks that it finished.

### 6. Check and open the result

**$\color{#FC0FC0}{Go\ back\ to\ your\ HOME\ directory, It's\ in\ your\ HOME\ directory}$**

You can open the fresh project folder in $\color{#FC0FC0}{VS\ Code}$, expand **results**, and click **0030_out.jpg**. Compare it with `inputs/0030.jpg`: the picture should show the same subject at a larger resolution.

**Use the exact filename `0030.jpg`, not `0030.png`.** An incorrect input path can make this script finish without producing an image.

## Evidence

The submitted images came from the successful fresh-clone run on **8 October 2026**, using Python 3.13.14 and a separate environment.

| File | Description |
| --- | --- |
| [evidence/input.jpg](evidence/input.jpg) | Copy of `inputs/0030.jpg`, 220 × 220 pixels. |
| [evidence/output.jpg](evidence/output.jpg) | Copy of `results/0030_out.jpg`, 880 × 880 pixels. |

Only this README and these two images belong in the submission. Do not add the working clone, `.venv`, model checkpoint or caches to `MLMPworkshop`.

## One real failure

**Category:** PYTHON VERSION.

**Problem:** Installing the local Real-ESRGAN project under Python 3.13 failed inside `setup.py`, in `get_version()`, with:

```text
KeyError: '__version__'
```

**Root cause:** The original function runs the version file using `exec()` and then looks for the result through `locals()`. That approach does not retrieve the newly created variable under Python 3.13.

**Minimal fix:** Replace the entire `get_version()` function with the version shown in Step 3. It creates a dictionary called `namespace`, runs the code using that dictionary, and reads `__version__` from it. Re-running the local installation then succeeded.

## AI agent check

**Which AI coding agent did you use?** OpenAI Codex in VS Code.

**What advice did it give?** Use the correct virtual environment, replace `get_version()` for Python 3.13, correct BasicSR's torchvision import, install the local project, and use the actual input filename `0030.jpg`.

**What command or fix did you choose to run yourself?** During class troubleshooting, I edited `setup.py` and ran the installation and inference commands myself. Later, I authorized Codex to perform the separate fresh-clone run and prepare the evidence and README. The fresh-run execution was agent-assisted.

**How did you verify the result?** I inspected the class result in VS Code. For the fresh run, Codex checked successful completion, opened the output, confirmed the size changed from 220 × 220 to 880 × 880, and visually inspected it. Step 6 lets the reviewer repeat those checks.

## Peer reproduction — to be completed by another student

Follow Steps 1–6 from a new clone using only this README, then record:

- **Reviewer:** Not yet recorded.
- **Reproduction result:** Not yet tested by a peer (PASS / FAIL).
- **Failed step, if any:**
- **Missing information, if any:**
- **Suggested fix, if any:**

Update these instructions after receiving the reviewer's feedback.
