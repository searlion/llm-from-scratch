## Setup

1.  **Prerequisites:**
    *   Git
    *   Python 3.9+
    *   [UV](https://github.com/astral-sh/uv) (Python package installer and resolver)
        *   Install UV: `pipx install uv` or `pip install uv`

2.  **After cloning the repository:**
    ```bash
    cd llm-from-scratch
    ```

3.  **Create virtual environment and install dependencies:**
    ```bash
    uv venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate  # Windows (cmd)
    # .venv\Scripts\Activate.ps1 # Windows (PowerShell)

    # Install dependencies from the lock file (recommended)
    uv pip sync
    ```

And that's it!