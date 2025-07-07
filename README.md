```markdown
# mefgraphs

A minimal implementation of learning graphs and their isomorphism classes
using Minimum Energy Flow (MEF) learning in McCulloch-Pitts neural networks

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd mefgraphs
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    conda create -n mefgraphs python=3.12
    conda activate mefgraphs
    ```

3.  **Install dependencies:**
    It's assumed you have a `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is not available, you might need to install common libraries for NeRF projects, such as:
    ```bash
    pip install numpy matplotlib ipython
    ```

## Usage

The primary script for making figures with small numbers of vertices V is `learning_graph_demo.py`.

**Example:**

```bash
ipython
run learning_graph_demo
```

