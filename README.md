# Vision World Models for Competitive Snake Playing

## Getting Started

Clone this repository to your local machine:

```bash
git clone https://github.com/gregorylimeurhen/snekboros.git
cd snekboros
```

### Environment Setup

This project was developed and tested on Python 3.13. It may work on other versions, but compatibility is not guaranteed.

#### Using `uv` (Recommended)

We recommend using [`uv`](https://docs.astral.sh/uv/), a fast Python package and project manager, written in Rust. You can install `uv` by following the installation instructions [here](https://docs.astral.sh/uv/getting-started/installation/). Alternatively, you can install via `pip`:

```bash
pip install uv
```

Then, simply synchronize your virtual environment using:

```bash
uv sync
```

Do note that PyTorch installation may be dependent on your CUDA version. Check your CUDA version with:

```bash
nvidia-smi
```

The `pyproject.toml` file has already been configured to target CUDA 12.8:

```toml
...
# For PyTorch installation
[tool.uv.sources]
torch = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
    { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

If you are targetting a different CUDA version (or CPU only), simply refer to the [uv documentation](https://docs.astral.sh/uv/guides/integration/pytorch/#installing-pytorch) and modify the `pyproject.toml` file accordingly.

---

#### Using `pip` and `venv` 

Alternatively, you can use `pip` and `venv`. You can create a virtual environment and install the required packages using:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## World Model Architecture (Work-in-Progress)
- **Observation Model (Vision Encoder):** The observation model encodes the raw image data of the current game state (i.e. image observation $o_t$) to its corresponding hidden representation (latent state $z_t$). This embedding captures the essential information about the current game (world) state, with unnecessary (or unpredictable) details filtered out (abstracted away). 
    - DINOv2/3: As proposed by DINO-WM, we can leverage the pretrained patch-features from the DINOv2/3 model, which provides both a spatial and object-representation world prior.  
        - $z_t\in\mathbb{R}^{N\times E}$, where $N$ denotes the number of patches and $E$ denotes the embedding dimension. 
        - By using a pretrained vision encoder and keeping it frozen during training (though we could choose to fine-tune it as well), we are essentialyl treating perception as a general task that benefits from large-scale internet data, and not something that needs to be learned from scratch when facing a new environment.
    - Custom Vision Encoder: Alternatively, we can also train a custom vision encoder from scratch in a self-supervised manner (i.e. JEPA). This allows us to obtain more task-specific hidden representations.
        - Given that the Snake game environment is relatively simple and low-dimension (16 x 16 grid), the vision encoder can be trained directly on this native resolution (as opposed to upsampling for DINO's 224 x 224 input), resulting in a much smaller observation model.  
        - For such low resolution images, even a 2 x 2 convolutional kernel might be too large, as it would already cover a significant portion of the image. We could use a simple ViT architecture with a patch size of 1 (i.e. each pixel is treated as a patch), allowing each pixel to attend to every other pixel in the image.
- **Transition Model (Dynamics Model):** The transition model learns to predict the next latent state $z_{t+1}$ given the current latent state $z_t$ and the action taken $a_t$ (or alternatively, given a history of latent states $z_{t-H:t}$ and actions $a_{t-H:t}$, where $H$ is the history length). This model captures the dynamics of the environment, allowing us to simulate future states based on current observations and actions.
    - A simple ViT would work (but without tokenization, since we already pass it in the form of patch features).
    - As proposed in DINO-WM, we can use a special attention mask such that each patch embedding $z_t^i$ for the latent state $z_t$ attends to $\{ z^i_{t-H:t-1}\}^N_{i=1}$ but **not** $\{ z^i_t \}^{<k}_{i=1}$. The idea is that treating patch vectors of one observation as a whole better captures temporal dynamics.
    - For actions, we can encode actions using an MLP to a higher dimensional representation e.g. $\phi(a_t)\in\mathbb{R}^K$ and then simply concatenate the action vector to each patch vector: $[z_t^i;\phi(a_t)]\in\mathbb{R}^{E+K}$.
- **Energy/Cost Module:** Outputs a scalar value measuring how "good" a particular state is. 
    - Intrinsic Cost (explicitly defined, non-learnable).
    - Trainable Cost.
    - Each can have multiple modules (representing different signals) and linearly combined. 
