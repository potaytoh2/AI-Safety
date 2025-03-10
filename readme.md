# Set-up
1. Create your virtual environment
`python -m venv venv`

2. Activate it cuh. Note the code below is for windows, not sure about mac
`venv/Scripts/Activate.ps1`

3. Install the requirements
`pip install -r requirements.txt`
Note that you only do this if you're using cpu to run this. If not you need to run:
`pip install transformers`
and the pytorch library for GPUs. 

4. Avoid using Jupyter files when training on GPU
- You can use the sbatchTemplatePython file to submit to the job scheduler on the SCIS GPU cluster. 
- Just ssh into your account on campus, git clone this repo
- modify the template python file then submit it.