import os
import subprocess
import sys

# Ensure we're in the right directory
os.chdir(r"c:\Users\aryav\OneDrive\Desktop\Documents\Projects\Projectt")

python_exe = sys.executable

# Use jupytext to convert notebook.py to notebook.ipynb
print("Converting notebook.py to notebook.ipynb...")
subprocess.run([python_exe, "-m", "jupytext", "--to", "notebook", "notebook.py"], check=True)

# Use jupyter nbconvert to execute the notebook inplace
print("Executing notebook.ipynb...")
subprocess.run([python_exe, "-m", "jupyter", "nbconvert", "--execute", "--to", "notebook", "--inplace", "notebook.ipynb"], check=True)
print("Done!")
