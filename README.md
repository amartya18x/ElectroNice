#ELECTRONICE

##Training
The main entry point to the code for training is trainmnist.py

##Layers
__electronice.py__ defines each individual layer. It has two classes one for the layer and the other generates the parameters for the layer.
__model_s.py__ is the new improved version of model.py. THis allows the use of a parameter num_layer where you can create n_layered network by simply putting num\_layer = n
__sample.py__ creates the Sampler class from model_s.py . THis generates a noise from a logistic distribution and outputs an image.
__test.py__ is used to test the network on a different set to see the log-likelihood on that set and also check for reconstruction etc.
