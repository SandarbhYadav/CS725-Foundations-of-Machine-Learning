README.TXT
---------------

TEAM NAME: Zero1

Team Members:
------------
Jimut Bahan Pal - 22D1594
Prateek Chanda - 22D0362
Sandarbh Yadav - 22D0374 
------------------------------------

SPECIAL FEATURES OF THE CODE
-----------------------------

Initially we were using list based operations for backprop and forward pass, later we found
that list are slow. We have have used numpy vectorizations for faster convergence and running 
of our code. 

We have implemented PCA (Principal Component Analysis) from scratch for  finding the best 
features in features selection part of the assignment. The selected features are stored in the 
file features.csv, in which about 78 components are selected.

NOTES FOR RUNNING THE CODE:

Please run the code using the following steps:
---------------------------------------------------

1. Make sure you are inside the directory 22D1594.

2. Please place the classification and regression data inside the 22D1594 folder.

3. Run the codes using the following commands for proper output
   $ python3 nn_1.py
   $ python3 nn_2.py
   $ python3 nn_3.py
   $ python3 classification/nn_classification.py

------------------------------------------------------

The code is highly vectorised so it is very fast. The execution times are given below:

nn_1.py: 1m 12s
nn_2.py: 1m 1s
nn_3.py: 1m 0s

Adam Optimizer and Principal Component Analysis are implemented from scratch too.

Extra credit Classification task is also done.

------------------------------------------------------