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

### Observation Model
At each time step $t$, the observation model encodes an image $o_t$ (raw image data of the current game state) to latent state (patch embeddings) $z_t\in\mathbb{R}^{N\times E}$, where $N$ denotes the number of patches and $E$ denotes the embedding dimension. These latent states are favourably low-dimensional, abstract representations of the game state that capture the essential information needed, while filtering out unnecessary details that do not aid prediction.

#### DINOv2 Vision Encoder
As proposed in the DINO-WM paper, we use the DINOv2 vision encoder as the observation model. DINOv2 is trained on large-scale internet data, hence providing a strong spatial and object-representation world prior. The DINOv2 vision encoder is kept frozen during training and testing. We are essentially treating perception as a general task that benefits from large-scale pretraining, and not something that needs to be learned from scratch when facing a new environment.
- DINOv2 expects 224 x 224 RGB images as input (i.e. $o_t\in\mathbb{R}^{3\times 224 \times 224}$), so we will need to upsample the original 16 x 16 grid image from the Snake game environment. This is not ideal, as it introduces redundant information and results in a much larger observation model than necessary. However, it allows us to leverage the powerful pretrained DINOv2 vision encoder without needing to train a custom vision encoder from scratch.
- $o_t$ can be encoded into either the patch token embeddings $z_t\in\mathbb{R}^{N\times E}$, or the `[CLS]` token embedding $z_t\in\mathbb{R}^{E}$ (though for implementation purposes we treat the `[CLS]` token as a single patch i.e. $z_t\in\mathbb{R}^{1\times E}$).
    - The `[CLS]` token aggregates information from all the patch tokens and serves as a global representation of the image, while the patch token embeddings preserve spatial information and represent local features of the image.

#### Alternatives
Alternatively, we can also train a custom vision encoder from scratch in a self-supervised manner (i.e. JEPA). This allows us to obtain more task-specific hidden representations. Given that the Snake game environment is relatively simple and low-dimension (16 x 16 grid), the vision encoder can be trained directly on this native resolution (as opposed to upsampling for DINO's 224 x 224 input), resulting in a much smaller observation model. For such low resolution images, even a 2 x 2 convolutional kernel might be too large, as it would already cover a significant portion of the image. We could use a simple ViT architecture with a patch size of 1 (i.e. each pixel is treated as a patch), allowing each pixel to attend to every other pixel in the image.

### Transition Model (Dynamics Model)
The transition model captures the dynamics of the environment, allowing us to simulate future states based on current observations and actions. The transition model takes in a history of past latent states $z_{t-H:t}$ and actions $a_{t-H:t}$, where $H$ is a hyperparameter that determines the history/context length, and predicts the latent state at the next time step $z_{t+1}$. 

We adopt a ViT architecture without the tokenization step, since we already have the patch embeddings (observation latent states) from the observation model. We flatten each latent state $z_t\in\mathbb{R}^{N\times E}$ into a sequence of patch embeddings $z_t^i\in\mathbb{R}^E$ for $i=1,...,N$, essentially treating each patch embedding as a token. As a result, the transition model receives a sequence of length $N\times H$ as input.

To capture temporal dependencies between the latent states, we use a causal attention mask. Specifically, each patch embedding $z_t^i$ for the latent state $z_t$ attends to $\{ z^i_{t-H:t-1}\}^N_{i=1}$ but **not** $\{ z^i_t \}^{<k}_{i=1}$, where $k$ is the index of the current patch embedding. The idea is that treating patch vectors of one observation as a whole better captures temporal dynamics, as opposed to allowing each patch embedding to attend to all other patch embeddings in the same time step.

As mentioned previously, the transition model also takes in a history of past actions $a_{t-H:t}$, which are crucial for predicting the next state. There are two approaches for passing in action information to the transition model. One way is to encode actions (e.g. using an MLP) to a higher dimensional representation i.e. $\phi(a_t)\in\mathbb{R}^K$, and then simply concatenate the action vector to each patch vector of the corresponding latent state. The concatenations $[z_t^i;\phi(a_t)]\in\mathbb{R}^{E+K}$ (for all patches and time steps) are passed into the ViT as tokens. This way, actions are treated as additional features for each patch embedding, allowing the model to learn how actions influence the state transitions. The alternate approach is to treat the encoded actions $\phi(a_t)\in\mathbb{R}^K$ as separate additional tokens. This requires the encoded actions to be in the same dimensional space as the patch embeddings (i.e. $K=E$) so that they can be concatenated along the sequence dimension and passed into the ViT. Any available propioceptive information can also be encoded and incorporated into the input in a similar manner.

Rather than predicting the next state $z_{t+1}$ by autoregressively decoding one patch embedding at a time, we simply use the ViT to encode the $N\times H$ sequence of patch embeddings (with action and propioceptive information incorporated via concatenation). Since the attention is causal, we can simply use the processed patch embeddings for the last time step $t$ and use them as the prediction for the next state $z_{t+1}$.

### Energy/Cost Module
The Energy/Cost Module takes in a state's latent representation $z_t$ and outputs a scalar value that measures how "good" the state is. This is akin to a value function in reinforcement learning, but instead of estimating expected future rewards, it estimates an intrinsic cost that captures how desirable a state is based on certain criteria. This module can be composed of several sub-modules, each representing a different signal that we want to capture in our cost function. For example:
- Intrinsic Cost (explicitly defined, non-learnable).
- Trainable Cost (learnable).
- Each can have multiple modules (representing different signals) and linearly combined. 

