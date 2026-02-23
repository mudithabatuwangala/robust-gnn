# robust-gnn

Graph Neural Network implementation (currently optimized for CPU).

## Setup Instructions

1. **Clone the Repo**: Run `git clone https://github.com/mudithabatuwangala/robust-gnn.git`.
2. **Enter the folder**: Run `cd robust-gnn`.
3. **Create Environment**: Run `python -m venv venv` to create your local environment.
4. **Activate Environment**:
   - Windows: `.\venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
     - Note: If you get an error saying `script execution is disabled on this system`,
       - Run this command first: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`. Then try the activation command again.
5. **Install Packages**: Run `pip install -r requirements.txt` to install Torch and PyG.
6. **Run Code**: Run `python main.py` to verify the installation.
