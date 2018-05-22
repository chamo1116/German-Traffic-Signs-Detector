import urllib.request
import zipfile
import click
import imageio
import pandas
import numpy as np
import six
import pickle
from PIL import Image
import numpy as np
import numpy, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import interactive
from os import listdir
import tensorflow as tf
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

#disables the warning, doesn't enable AVX/FMA for CPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



@click.command()

@click.option('-m',default='',help='choose the model: LRkit or LRtensor or lenet')
@click.option('-d',default='',help="PATH where is the image dataset i.e C:/blabla/blabla only with / not with")
@click.argument('command')
def data(command,m,d):
        if command == 'download':
                click.echo('Downloading Dataset ---')
                response = urllib.request.urlretrieve("http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip", "dataset.zip")
	
                with zipfile.ZipFile('dataset.zip',"r") as z:
                        z.extractall("E:/German-Traffic-Signs-Detector/images/train")
                        
        elif command=="train":
                if m=='LRscikit' and d!='':
                        click.echo("Training with logistic regression using scikit...")
                        Xlist,Ylist=processimage(d)
                        x_train, x_test, y_train, y_test = train_test_split(Xlist, Ylist, test_size=0.20, random_state=0)
                        accuracy=trainLRscikit(x_train,x_test,y_train,y_test)
                        print ("Accuracy: "+str(accuracy))
                if m=='LRtensor' and d!='':
                        click.echo("Training with logistic regression using tensor flow...")
                        Xlist,Ylist=processimage(d)
                        #We need onehot encode Ylist for the model (dimensions)
                        Ylist=Onehot_encode(Ylist)
                        x_train, x_test, y_train, y_test = train_test_split(Xlist, Ylist, test_size=0.20, random_state=0)
                        accuracy= trainLRtensor(x_train, x_test, y_train, y_test)
                        print("Accuracy: "+str(accuracy*100))
                if m=='lenet' and d!='':
                        click.echo("Training a CNN with lenet and tensorflow...")
                        Xlist,Ylist=processimageb(d)
                        Ylist=Onehot_encode(Ylist)
                        x_train, x_test, y_train, y_test = train_test_split(Xlist, Ylist, test_size=0.20, random_state=0)
                        accuracy=trainlenet(x_train,x_test,y_train,y_test)
                        print("Accuracy: "+str(accuracy*100))
                        
        elif command=="test":
                if m=='LRscikit' and d!='':
                        click.echo("Testing with logistic regression using scikit...")
                        img,label=processimage(d)
                        img_train, img_test, label_train, label_test = train_test_split(img, label, test_size=0.99, random_state=0)
                        loaded_model = pickle.load(open("models/model1/saved/LRscikitmodel.sav", 'rb'))
                        result = loaded_model.score(img_test, label_test)
                        accuracy=result*100
                        print ("Accuracy: "+str(accuracy))
                if m=='LRtensor' and d!='':
                        click.echo("Testing with logistic regression using tensorflow...")
                        Xlist,Ylist=processimage(d)
                        #We need onehot encode Ylist for the model (dimensions)
                        Ylist=Onehot_encode(Ylist)
                        accuracy=testLRtensor(Xlist,Ylist)
                        print("Accuracy: "+str(accuracy*100))
                if m=='lenet' and d!='':
                        click.echo("Testing CNN with lenet and tensorflow...")
                        Xlist,Ylist=processimageb(d)
                        Ylist=Onehot_encode(Ylist)
                        accuracy=testlenet(Xlist,Ylist)
                        print("Accuracy: "+str(accuracy*100))
        elif command=="infer":
                if m=='LRscikit' and d!='':
                        Xlist,Ylist=processimage(d)
                        loaded_model = pickle.load(open("models/model1/saved/LRscikitmodel.sav", 'rb'))
                        inferscikit(d,Xlist,Ylist,loaded_model)
                if m=='LRtensor' and d!='':
                        Xlist,Ylist=processimage(d)
                        Xlist = np.array(Xlist, dtype='float32')
                        Xlistr = np.reshape(Xlist, (Xlist.shape[0], -1))
                        infertensor(d,Xlistr,Ylist)
                if m=='lenet' and d!='':
                        Xlist,Ylist=processimageb(d)
                        inferlenet(d,Xlist,Ylist)
        else:
                click.echo("Missing arguments")
                        
                                
def processimage(path):
        path=path+"/"
        Xlist=[]
        Ylist=[]
        #Custom size
        width = 35
        height = 35
        #Read the image from the directory
        for directory in os.listdir(path):
                #Filter the images we don't need
                if len(directory)==2:
                        for file in os.listdir(path+"/"+directory):
                            #Open the image    
                            img=Image.open(path+directory+"/"+file)
                            #Resize the image to custom size
                            img = img.resize((width, height))
                            #Convert to an array the image
                            featurevector = numpy.array(img.getdata())
                            #Add in two lists, Xlist with the image data and Ylist with the labels
                            Xlist.append(featurevector)
                            Ylist.append(directory)
        Xlist = np.array(Xlist, dtype='float32')
        #Convert in two dimensions because train_test_split need an array as maximum two dimensions
        Xlistr = np.reshape(Xlist, (Xlist.shape[0], -1))
        return Xlistr,Ylist

def processimageb(path):
        path=path+"/"
        Xlist=[]
        Ylist=[]
        data=[]
        width = 32
        height = 32
        for directory in os.listdir(path):
                for file in os.listdir(path+directory):
                    img=Image.open(path+directory+"/"+file)
                    img = img.resize((width, height)) 
                    featurevector = numpy.array(img)
                    Xlist.append(featurevector)
                    Ylist.append(directory)
        Xlist = np.array(Xlist, dtype='float32')
        return Xlist,Ylist


def trainLRscikit(x_train,x_test,y_train,y_test):

        #Create the model
        logisticRegr = LogisticRegression(solver = 'lbfgs')
        #Train the model
        logisticRegr.fit(x_train, y_train)
        # save the model 
        model = 'LRscikitmodel.sav'
        pickle.dump(logisticRegr, open("models/model1/saved/"+model, 'wb'))
        #Calculate the accuracy
        score = logisticRegr.score(x_test, y_test)
        accuracy=score*100
        return accuracy

                

def inferscikit(path,x_test,y_test,loaded_model):
        path=path+"/"
        total= numberfiles(path)
        width = 35
        height = 35
        cont=1
        #Read the image from the directory
        for directory in os.listdir(path):
                #Filter the images we don't need
                if len(directory)==2:
                        for file in os.listdir(path+"/"+directory):
                            #Open the image    
                            img=Image.open(path+directory+"/"+file)
                            #Resize the image to custom size
                            img = img.resize((width, height))
                            #Predict the label of each img with scikit
                            prediction=loaded_model.predict(x_test[cont-1].reshape(1,-1))
                            plt.figure(cont)
                            plt.title("Prediction Label "+str(prediction[0]+" Real label "+ y_test[cont-1]), fontsize = 12)
                            imgplot = plt.imshow(img)
                            #Hold the images 
                            if cont < total:
                                     interactive(True)
                            else:
                                     interactive(False)
                            plt.show()
                            cont=cont+1
def trainLRtensor(x_train,x_test,y_train,y_test):
        ntrain=len(x_train)
        ntest=len(x_test)
        dim= x_train.shape[1]
        Totallabel=y_train.shape[1]
        
        # Parameters of Logistic Regression
        learning_rate   = 0.001
        training_epochs = 1000
        batch_size      = 10
        display_step    = 100

        x = tf.placeholder("float", [None, dim], name="x")
        y = tf.placeholder("float", [None, Totallabel], name= "y")
              
        #Set model weights
        W = tf.Variable(tf.zeros([dim, Totallabel]))
        b= tf.Variable(tf.zeros([Totallabel]))

        #Linear mapping
        prediction = tf.matmul(x, W) + b
        model = tf.nn.softmax(tf.matmul(x, W) + b,name="model")

        #cost function
        cost_function = -tf.reduce_sum(y*tf.log(prediction))
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2( logits=prediction, labels=y)
        loss = tf.reduce_mean(entropy) # computes the mean over examples in the batch

        #Optimizer
        optm = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                num_batch = int(ntrain/batch_size)
                # Loop over all batches
                for i in range(num_batch): 
                    randidx = np.random.randint(ntrain, size=batch_size)
                    batch_xs = x_train[randidx, :]
                    batch_ys = y_train[randidx, :]
                    # Fit training using batch data
                    sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
                    # Compute average loss
                    avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/num_batch
            predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1), name="predictions")
            accuracy = tf.reduce_mean(tf.cast(predictions, "float"),name="accuracy")
            Accuracy= accuracy.eval({x: x_test, y: y_test})
            # Save the variables to disk.
            save_path = saver.save(sess, "models/model2/saved/LRtensor")

            sess.close()
        return Accuracy

def testLRtensor(x_test,y_test):
        inference_graph = tf.Graph()
        with tf.Session(graph= inference_graph) as sess:
            loader = tf.train.import_meta_graph('models/model2/saved/LRtensor.meta')
            loader.restore(sess, 'models/model2/saved/LRtensor')
            _accuracy = inference_graph.get_tensor_by_name('accuracy:0')
            _x  = inference_graph.get_tensor_by_name('x:0')
            _y  = inference_graph.get_tensor_by_name('y:0')
            accuracy= _accuracy.eval({_x: x_test , _y: y_test})
            sess.close()
        return accuracy

def infertensor(path,x_test,y_test):
        path=path+"/"
        total= numberfiles(path)
        width = 35
        height = 35
        cont=1
        #Read the image from the directory
        for directory in os.listdir(path):
                #Filter the images we don't need
                if len(directory)==2:
                        for file in os.listdir(path+"/"+directory):
                            #Open the image    
                            img=Image.open(path+directory+"/"+file)
                            #Resize the image to custom size
                            img = img.resize((width, height))
                            x_partial=[x_test[cont-1]]
                            #Predict the label of each img with tensor
                            prediction=predict(x_partial)
                            plt.figure(cont)
                            plt.title("Prediction Label: "+"0"+str(prediction[0]) + " Real label: "+ str(y_test[cont-1]), fontsize = 12)
                            imgplot = plt.imshow(img)
                            #Hold the images 
                            if cont < total:
                                     interactive(True)
                            else:
                                     interactive(False)
                            plt.show()
                            cont=cont+1
def trainlenet(x_train,x_test,y_train,y_test):
        #Set the number of channels
        channels= x_train.shape[3]
        #Set the number of labels
        labels=y_train.shape[1]

        #Set up tensorflow
        EPOCHS = 50
        BATCH_SIZE = 128
        rate = 0.001
        #Features and Labels
        x = tf.placeholder(tf.float32, (None, 32, 32, channels), name="x")
        y = tf.placeholder(tf.float32, (None,labels), name="y")

        #Training Pipeline
        logits = LeNet(x,channels,labels)
        #Model for prediction
        model = tf.nn.softmax(logits,name="model")
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = rate)
        training_operation = optimizer.minimize(loss_operation)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")
        #Train the Model
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(x_train)
            for i in range(EPOCHS):
                #Choose random data
                x_train, y_train = shuffle(x_train, y_train)
                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                    sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            Accuracy=accuracy_operation.eval({x: x_test, y: y_test})
            save_path = saver.save(sess, "models/model3/saved/lenet")
        sess.close()
        return Accuracy

def testlenet(x_test,y_test):
        inference_graph = tf.Graph()
        with tf.Session(graph= inference_graph) as sess:
            loader = tf.train.import_meta_graph('models/model3/saved/lenet.meta')
            loader.restore(sess, 'models/model3/saved/lenet')
            _x  = inference_graph.get_tensor_by_name('x:0')
            _y  = inference_graph.get_tensor_by_name('y:0')
            _accuracy = inference_graph.get_tensor_by_name('accuracy:0')
            accuracy= _accuracy.eval({_x: x_test , _y: y_test})
            sess.close()
        return accuracy

def inferlenet(path,x_test,y_test):
        path=path+"/"
        total= numberfiles(path)
        width = 32
        height = 32
        cont=1
        #Read the image from the directory
        for directory in os.listdir(path):
                #Filter the images we don't need
                if len(directory)==2:
                        for file in os.listdir(path+"/"+directory):
                            #Open the image    
                            img=Image.open(path+directory+"/"+file)
                            #Resize the image to custom size in this case 32x32
                            img = img.resize((width, height))
                            x_partial=[x_test[cont-1]]
                            prediction=predictlenet(x_partial)
                            plt.figure(cont)
                            plt.title("Prediction Label: "+"0"+str(prediction[0]) + " Real label: "+ str(y_test[cont-1]), fontsize = 12)
                            imgplot = plt.imshow(img)
                            if cont < total:
                                     interactive(True)
                            else:
                                     interactive(False)
                            plt.show()
                            cont=cont+1

def numberfiles(path):
        globalcount=0
        for directory in os.listdir(path):
              globalcount=globalcount+len(listdir(path+directory))
        return globalcount
def Onehot_encode(Ylist):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(Ylist)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_Ylist = onehot_encoder.fit_transform(integer_encoded)
        return onehot_Ylist
              

def predict(x_test):
        inference_graph = tf.Graph()
        with tf.Session(graph= inference_graph) as sess:
            loader = tf.train.import_meta_graph('models/model2/saved/LRtensor.meta')
            loader.restore(sess, 'models/model2/saved/LRtensor')
            _model  = inference_graph.get_tensor_by_name('model:0')
            _x  = inference_graph.get_tensor_by_name('x:0')
            prediction = tf.argmax(_model, 1)
            prediction=prediction.eval(feed_dict={_x: x_test})
            return prediction
def LeNet(x,channels,labels):    
        # Hyperparameters
        mu = 0
        sigma = 0.1
        layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16,
        'layer_3' : 120,
        'layer_f1' : 43
        }


        # TODO: Layer 1: Convolutional. Input = 32x32xchannels. in our case 3 Output = 28x28x6.
        conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,channels,6],mean = mu, stddev = sigma))
        conv1_b = tf.Variable(tf.zeros(6))
        conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
        # TODO: Activation.
        conv1 = tf.nn.relu(conv1)

        # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
        pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

        # TODO: Layer 2: Convolutional. Output = 10x10x16.
        conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
        # TODO: Activation.
        conv2 = tf.nn.relu(conv2)

        # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
        pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 

        # TODO: Flatten. Input = 5x5x16. Output = 400.
        fc1 = flatten(pool_2)

        # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
        fc1_b = tf.Variable(tf.zeros(120))
        fc1 = tf.matmul(fc1,fc1_w) + fc1_b

        # TODO: Activation.
        fc1 = tf.nn.relu(fc1)

        # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
        fc2_b = tf.Variable(tf.zeros(84))
        fc2 = tf.matmul(fc1,fc2_w) + fc2_b
        # TODO: Activation.
        fc2 = tf.nn.relu(fc2)

        # TODO: Layer 5: Fully Connected. Input = 84. Output = num labels.
        fc3_w = tf.Variable(tf.truncated_normal(shape = (84,labels), mean = mu , stddev = sigma))
        fc3_b = tf.Variable(tf.zeros(labels))
        logits = tf.matmul(fc2, fc3_w) + fc3_b
        return logits

def predictlenet(x_test):
        inference_graph = tf.Graph()
        with tf.Session(graph= inference_graph) as sess:
            loader = tf.train.import_meta_graph('models/model3/saved/lenet.meta')
            loader.restore(sess, 'models/model3/saved/lenet')
            _model  = inference_graph.get_tensor_by_name('model:0')
            _x  = inference_graph.get_tensor_by_name('x:0')
            prediction = tf.argmax(_model, 1)
            prediction=prediction.eval(feed_dict={_x: x_test})
            return prediction
        
if __name__== '__main__':
	data()


