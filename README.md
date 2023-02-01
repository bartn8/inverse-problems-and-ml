# inverse-problems-and-ml
Exam of module "Inverse problems and Machine learning" from PhD winter school "Advanced Methods for Mathematical Image Analysis"

1. Setting up the environment
Install anaconda by following official installation instructions at https://www.anaconda.com/products/individual

It should look similar to the following commands, but with different software versions  

   $ wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
   $ chmod  +x Anaconda3-5.2.0-Linux-x86_64.sh
   $ ./Anaconda3-5.2.0-Linux-x86_64.sh

2. Create a new environment named "myenv" by 

   $ conda create -n myenv python=3.10

3. Activate this environment

   $ conda activate myenv

4. Install PyTorch 

   $ conda install -c pytorch pytorch torchvision

5. Install ASTRA and SciKit-image back-ends for ODL (needed for doing tomography)

   $ conda install -c astra-toolbox/label/dev astra-toolbox
   $ conda install -c anaconda scikit-image

6. Install ODL from source

   $ git clone https://github.com/odlgroup/odl
   $ cd odl
   $ pip install --editable .

7. Install other packages 

   $ conda install matplotlib jupyter ipykernel

8. Install jupyter and make your environment visible in the jupyter notebook

   $ python -m ipykernel install --user --name myenv --display-name="My Env"
