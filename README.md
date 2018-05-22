"# German-Traffic-Signs-Detector" 
Here you can find differents methods for clasify German Traffic Signs. Save in image folder your dataset, train and test dataset respectively. It will change the size of images to 35x35 but for lenet we resize to 32x32 


Model 1: Use a logistic regression with scikit learn of python

Train: 
python app.py train -m LRscikit -d C:/blabla/blabla

Note: The path is with "/"

Test:
python app.py test -m LRscikit -d [directory with test data]

Infer:
python app.py infer -m LRscikit -d [directory with user data]

It will plot the images with predict and real label respectively


Model 2: 

Train:
python app.py train -m LRtensor -d [directory with trainig data]

Test:
python app.py test -m LRtensor -d [directory with test data]


Infer:
python app.py infer -m LRtensor -d [directory with user data]


Model 3: 

Train:
python app.py train -m lenet -d [directory with trainig data]

Test:
python app.py test -m lenet -d [directory with test data]


Infer:
python app.py infer -m lenet -d [directory with user data]


