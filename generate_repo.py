import os

# -----------------------------
# Folder + file definitions
# -----------------------------
structure = {
    "fractal-memory-ai": {
        "README.md": "# Fractal Memory AI\n\nA multi-timescale continual learning demo.\n",
        "requirements.txt": "torch\nnumpy\nmatplotlib\n",
        ".gitignore": "venv/\n__pycache__/\n*.pyc\n",
        
        "src": {
            "__init__.py": "",
            "data.py": "# synthetic signal generators\n",
            "reservoir.py": "# reservoir (fast memory)\n",
            "episodic.py": "# episodic buffer (medium memory)\n",
            "readout.py": "# fast readout + consolidation (slow memory)\n",
            "fractal_model.py": "# links reservoir + episodic + readout into a full model\n",
            "utils.py": "# misc utilities\n"
        },
        
        "experiments": {
            "demo_fractal_memory.py": "# main runnable demo script\n",
            "ablation_reservoir_only.py": "# experiment: reservoir only\n",
            "ablation_no_consolidation.py": "# experiment: no slow consolidation\n",
            "config.yaml": "learning_rate: 0.0001\nreservoir_size: 300\nconsolidate_every: 400\n"
        },

        "notebooks": {
            "exploratory_plots.ipynb": "",
            "reservoir_dynamics.ipynb": ""
        },

        "results": {
            "plots": {},
            "logs": {
                "run1.csv": "step,pred,true\n"
            }
        },

        "models": {
            "saved_readout.pth": ""
        }
    }
}

# -----------------------------
# Recursive folder builder
# -----------------------------
def build(base, tree):
    for name, content in tree.items():
        path = os.path.join(base, name)

        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            build(path, content)
        else:
            # Create file with placeholder content
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

# -----------------------------
# Build the repository
# -----------------------------
root = list(structure.keys())[0]
print(f"Creating repo structure: {root}")
build(".", structure)
print("Done! Folder structure created successfully.")
