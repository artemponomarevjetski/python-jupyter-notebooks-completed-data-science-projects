A large collection of data science models in Python, plain files and Jupyter notebooks.

To prepare the code for execution on a Mac do this
% cd Desktop/ML_task/
% python3 -m venv  .
% source ./venv activate
% pip3 install --upgrade pip
% pip3 install -r requirements.txt
# if the above does not work, install the libraries individually
% pip3 install numpy
% pip3 install pandas 
% python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
% pip3 install opencv-python
# use command 'python3 train_v2.py -f data', or with more flags; the new flag, -a, is to choose between 'SGD' and 'Adam' optimizer
% source ./venv deactivate
