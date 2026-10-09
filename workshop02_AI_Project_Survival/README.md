# Workshop 02 — AI Project Survival

## Project

- **Repository:** https://github.com/xinntao/Real-ESRGAN
- **Revision:** `a4abfb2979a7bbff3f69f58f58ae324608821e27`
- **Inference task:** Use pretrained `RealESRGAN_x4plus` to upscale the included `0030.jpg` by 4× in each dimension. No training or face enhancement.
- **Reproduction date:** 8 October 2026.

The repository was cloned again into `workshop02_projects/Real-ESRGAN_fresh`, separate from the class copy in `Real-ESRGAN`. A new `.venv` was created inside the fresh clone; neither class environment (`.venv` or `.venv313`) was reused. uv's download/build cache was initially available. To remove reliance on the cached BasicSR wheel, BasicSR was subsequently built from its pinned source archive with the compatibility fixes below, and inference was repeated.

## Repo map

| Item | Location / purpose |
| --- | --- |
| Environment | Linux x86_64; uv 0.11.22; CPython 3.13.14; fresh clone's `.venv`; CPU inference (`torch.cuda.is_available()` was false) |
| Entry point | `inference_realesrgan.py`; its `main()` parses arguments, loads weights, reads images and writes results |
| Model | `RealESRGAN_x4plus`: RRDBNet, 23 blocks, 64 features, native 4× scale; inference helper in `realesrgan/utils.py` |
| Checkpoint | `weights/RealESRGAN_x4plus.pth`, from the upstream v0.1.0 release |
| Input | `inputs/0030.jpg`, a 220×220 RGB illustration included in the repository |
| Output | `results/0030_out.jpg`, 880×880 RGB JPEG |
| Setup | `requirements.txt` declares dependencies; `setup.py` generates `realesrgan/version.py` and installs the local project |

## Environment setup

Run the following in a terminal. This recipe assumes `uv`, Git and curl are available. Choose an unused folder name if `Real-ESRGAN_fresh` already exists. All later commands run from the fresh clone, with its environment active.

```bash
cd ~/Desktop/MLMPworkshop/workshop02_projects
git clone https://github.com/xinntao/Real-ESRGAN.git Real-ESRGAN_fresh
cd Real-ESRGAN_fresh
git checkout a4abfb2979a7bbff3f69f58f58ae324608821e27
uv python install 3.13.14
uv venv --python 3.13.14 .venv
source .venv/bin/activate
python --version
```

On the reproduction machine Python 3.13.14 was already installed, so `uv venv --python 3.13 .venv` selected it. The explicit patch version above makes that selection repeatable.

### Pin the dependencies

The first fresh installation used `uv pip install --python .venv/bin/python -r requirements.txt`. Because upstream requirements have few version limits, the following is the exact resolved dependency snapshot, subsequently checked with uv. Save it using this terminal block. BasicSR and Real-ESRGAN are installed separately from patched source below.

```bash
cat > /tmp/realesrgan-dependencies.txt <<'PINS'
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
uv pip install --python .venv/bin/python -r /tmp/realesrgan-dependencies.txt
```

These pins record this Linux environment, including PyTorch's CUDA dependencies even though inference ran on CPU. They are not a claim of portability to every operating system.

### Apply the Python 3.13 and torchvision compatibility fixes

Download the exact BasicSR source release. Keeping the source outside `.venv` makes the patch explicit and avoids depending on a previously edited cached wheel.

```bash
curl -fL https://files.pythonhosted.org/packages/86/41/00a6b000f222f0fa4c6d9e1d6dcc9811a374cabb8abb9d408b77de39648c/basicsr-1.4.2.tar.gz -o /tmp/basicsr-1.4.2.tar.gz
```

Paste this **whole block** into the terminal, including `python - <<'PY'` and the final `PY`. It checks the archive, extracts it safely, fixes version lookup in both setup scripts, and changes BasicSR's grayscale import to the public torchvision API.

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import tarfile

archive = Path('/tmp/basicsr-1.4.2.tar.gz')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == 'b89b595a87ef964cda9913b4d99380ddb6554c965577c0c10cb7b78e31301e87'
with tarfile.open(archive) as tf:
    tf.extractall('vendor', filter='data')

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
source = path.read_text()
source = source.replace(
    'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
    'from torchvision.transforms.functional import rgb_to_grayscale')
path.write_text(source)
PY
uv pip install --python .venv/bin/python --no-deps ./vendor/basicsr-1.4.2
uv pip install --python .venv/bin/python --no-deps -e .
```

The final command installs the fresh project in editable mode and generates `realesrgan/version.py`. `--no-deps` preserves the explicitly installed dependency versions. Build isolation may fetch build tools separately; the runtime versions are pinned above.

### Download the pretrained checkpoint

```bash
curl -fL https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -o weights/RealESRGAN_x4plus.pth
```

Checkpoint SHA-256: `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1`.

## Inference

Exact command executed in the fresh clone:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs/0030.jpg -o results --outscale 4 --fp32
```

`--fp32` selects full precision. The two environment variables limit CPU threads. `-i` names one actual JPEG, not the entire mixed-input folder. `--face_enhance` is omitted because this task needs only the Real-ESRGAN checkpoint. The script chooses the CPU when CUDA is unavailable.

Successful terminal output:

```text
Testing 0 0030
```

The process exited with code 0. The generated image is `results/0030_out.jpg`; `_out` is the script's default suffix and it retains the input extension.

## Evidence and verification

| Evidence | Source in fresh clone | Dimensions | Format |
| --- | --- | --- | --- |
| [Input](evidence/input.jpg) | `inputs/0030.jpg` | 220×220 | RGB JPEG |
| [Output](evidence/output.jpg) | `results/0030_out.jpg` | 880×880 | RGB JPEG |

Both images were opened successfully with Pillow. The output dimensions were checked to be exactly four times the input dimensions, and the generated output was visually inspected. This verifies a completed inference and valid output, not a quantitative restoration-quality score. There is no high-resolution ground truth in this submission.

![Input image](evidence/input.jpg)

![Real-ESRGAN output](evidence/output.jpg)

SHA-256 checksums of the submitted evidence:

```text
2f49c8cf6249940408fff736f394a4f8a228a6488bcc2fe9f92bdfb90b54a504  evidence/input.jpg
df1c6672dd9a6505237696e659cf8bd151ecc64bbc2c88dd0066ae4428f11898  evidence/output.jpg
```

Numerical and JPEG differences can occur with other hardware or library versions; the checksums identify these submitted artifacts.

## One real failure

**Category:** Python-version / package-build compatibility.

**Observed command:** `uv pip install --python .venv/bin/python --no-deps -e .` on the unmodified fresh clone under Python 3.13.14.

**Observed error:**

```text
File "<string>", line 86, in <module>
File "<string>", line 72, in get_version
KeyError: '__version__'
```

**Root cause:** `setup.py` executes the version file and then expects `locals()['__version__']` to contain the new variable. That implicit-locals pattern fails under Python 3.13.

**Minimal fix:** Pass an explicit dictionary to `exec()` and read `__version__` from that same dictionary, as shown in the setup block. The editable installation then succeeded. The same fix is applied to BasicSR's source setup code to avoid its equivalent build problem on a machine with no cached wheel.

**Other class troubleshooting:** BasicSR imported the removed `torchvision.transforms.functional_tensor` module; the reproducible setup uses `torchvision.transforms.functional` instead. An earlier whole-folder run with `--face_enhance` failed on grayscale input with shapes `(1404, 2000, 1)` versus `(1404, 2000)`. A single RGB input without face enhancement avoids that separate path. Specifying `0030.png` also produced no result because the repository's file is `0030.jpg`.

## AI agent check

**Which AI coding agent did you use?** OpenAI Codex in the IDE.

**What advice did it give?** It helped identify the active Python environment, explain that `cv2` comes from `opencv-python`, fix BasicSR's torchvision import and Python 3.13 version lookup, install the local project to generate `realesrgan/version.py`, and select the correct JPEG path. Its initial explanation for a failed-then-successful BasicSR installation remained uncertain; the original log did not establish why the retry succeeded.

**What command or fix did you choose to run yourself?** During class troubleshooting I ran the installation and inference commands in my terminal and edited the suggested compatibility code. After a failed whole-folder face-enhancement run, I used the single-image `0030.jpg` inference command and confirmed success in the conversation. For this final assignment, I explicitly authorized Codex to create the fresh clone, build the separate environment, repeat inference, verify the artifacts and write this report. The fresh-run commands were executed by the agent, not claimed as manual student execution.

**How did you verify the result?** In class I inspected the saved result in VS Code. For the fresh reproduction, Codex checked exit code 0, opened the files, confirmed 220×220 → 880×880 RGB dimensions, visually inspected the output, and recorded hashes. The submitted output comes from the fresh run, not the class results folder.

## Submission scope

```text
workshop02_AI_Project_Survival/
├── README.md
└── evidence/
    ├── input.jpg
    └── output.jpg
```

Only these three files belong in the submission. The fresh clone, environments, downloaded source, checkpoint, datasets and package/model caches remain outside this folder. When committing, add this folder explicitly; do not use `git add .` from the parent repository containing the working clones.

## Repository references

- [Upstream README at the reproduced revision](https://github.com/xinntao/Real-ESRGAN/blob/a4abfb2979a7bbff3f69f58f58ae324608821e27/README.md)
- [Inference script at that revision](https://github.com/xinntao/Real-ESRGAN/blob/a4abfb2979a7bbff3f69f58f58ae324608821e27/inference_realesrgan.py)
- [Project setup script at that revision](https://github.com/xinntao/Real-ESRGAN/blob/a4abfb2979a7bbff3f69f58f58ae324608821e27/setup.py)
