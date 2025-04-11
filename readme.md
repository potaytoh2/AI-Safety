# AI-Safety Project

## Setup instructions

1. Create a virtual environment using 'venv':

```sh
python3 -m venv .venv
```

> [!IMPORTANT]
> Please name the virtual environment to .venv so that it doesn't get version controlled

2. Activate virtual environment:

- On MacOS/Linux
```sh
source .venv/bin/activate
```

- On Windows
```sh
venv/Scripts/Activate.ps1
```

3. Install required Python packages:

- For CPU Users:
    - Install all packages specified in the 'requirements.txt' file by running:
```sh
pip install -r requirements.txt
```
> [!NOTE]
> The `requirements.txt` file was created from a Python 3.11 environment

- For GPU Users:
    - If you plan to run the project using GPU, skip the previous step and instead install the following:
```sh
pip install transformers
```

and the pytorch library for GPUs. 

4. Avoid using Jupyter files when training on GPU
- You can use the sbatchTemplatePython file to submit to the job scheduler on the SCIS GPU cluster. 
- Just ssh into your account on campus, git clone this repo
- modify the template python file then submit it.
