#!/usr/bin/env python
# coding: utf-8

# # First things first
# * Click **File -> Save a copy in Drive** and click **Open in new tab** in the pop-up window to save your progress in Google Drive.
# * Click **Runtime -> Change runtime type** and select **GPU** in Hardware accelerator box to enable faster GPU training.

# # Final project: Finding the suspect

#Author: SHIRSHAKK PURKAYASTHA

# <a href="https://en.wikipedia.org/wiki/Facial_composite">Facial composites</a> are widely used in forensics to generate images of suspects. Since victim or witness usually isn't good at drawing, computer-aided generation is applied to reconstruct the face attacker. One of the most commonly used techniques is evolutionary systems that compose the final face from many predefined parts.
# 
# In this project, we will try to implement an app for creating a facial composite that will be able to construct desired faces without explicitly providing databases of templates. We will apply Variational Autoencoders and Gaussian processes for this task.
# 
# The final project is developed in a way that you can apply learned techniques to real project yourself. We will include the main guidelines and hints, but a great part of the project will need your creativity and experience from previous assignments.

# ### Setup
# Load auxiliary files and then install and import the necessary libraries.

# In[ ]:


try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False
if IN_COLAB:
    print("Downloading Colab files")
    get_ipython().system(' shred -u setup_google_colab.py')
    get_ipython().system(' wget https://raw.githubusercontent.com/hse-aml/bayesian-methods-for-ml/master/setup_google_colab.py -O setup_google_colab.py')
    import setup_google_colab
    setup_google_colab.load_data_final_project()


# In[ ]:


get_ipython().system(' pip install GPy gpyopt')


# In[ ]:





# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow as tf
import GPy
import GPyOpt
import keras
from keras.layers import Input, Dense, Lambda, InputLayer, concatenate, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Deconv2D
from keras.losses import MSE
from keras.models import Model, Sequential
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.utils import np_utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import utils
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Grading
# As some of the final project tasks can be graded only visually, the final assignment is graded using the peer-review procedure. You will be asked to upload your Jupyter notebook on the web and attach a link to it in the submission form. Detailed submission instructions and grading criterions are written at the end of this notebook.

# ## Model description
# We will first train variational autoencoder on face images to compress them to low dimension. One important feature of VAE is that constructed latent space is dense. That means that we can traverse the latent space and reconstruct any point along our path into a valid face.
# 
# Using this continuous latent space we can use Bayesian optimization to maximize some similarity function between a person's face in victim/witness's memory and a face reconstructed from the current point of latent space. Bayesian optimization is an appropriate choice here since people start to forget details about the attacker after they were shown many similar photos. Because of this, we want to reconstruct the photo with the smallest possible number of trials.

# ## Generating faces

# For this task, you will need to use some database of face images. There are multiple datasets available on the web that you can use: for example, <a href="http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html">CelebA</a> or <a href="http://vis-www.cs.umass.edu/lfw/">Labeled Faces in the Wild</a>. We used Aligned & Cropped version of CelebA that you can find <a href="https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADSNUu0bseoCKuxuI5ZeTl1a/Img?dl=0&preview=img_align_celeba.zip">here</a> to pretrain VAE model for you. See optional part of the final project if you wish to train VAE on your own.

# <b>Task 1:</b> Train VAE on faces dataset and draw some samples from it. (You can use code from previous assignments. You may also want to use convolutional encoders and decoders as well as tuning hyperparameters)

# In[ ]:


sess = tf.InteractiveSession()
K.set_session(sess)


# In[ ]:


latent_size = 8


# In[ ]:


vae, encoder, decoder = utils.create_vae(batch_size=128, latent=latent_size)
sess.run(tf.global_variables_initializer())
vae.load_weights('CelebA_VAE_small_8.h5')


# In[ ]:


K.set_learning_phase(False)


# In[ ]:


latent_placeholder = tf.placeholder(tf.float32, (1, latent_size))
decode = decoder(latent_placeholder)


# #### GRADED 1 (3 points): Draw 25 samples from trained VAE model
# As the first part of the assignment, you need to become familiar with the trained model. For all tasks, you will only need a decoder to reconstruct samples from a latent space.
# 
# To decode the latent variable, you need to run ```decode``` operation defined above with random samples from a standard normal distribution.

# In[ ]:


### TODO: Draw 25 samples from VAE here
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    image = sess.run(decode, feed_dict={latent_placeholder: np.random.randn(1, latent_size)})[0] ### YOUR CODE HERE
    plt.imshow(np.clip(image, 0, 1))
    plt.axis('off')


# In[ ]:


print(decoder.summary())


# ## Search procedure

# Now that we have a way to reconstruct images, we need to set up an optimization procedure to find a person that will be the most similar to the one we are thinking about. To do so, we need to set up some scoring utility. Imagine that you want to generate an image of Brad Pitt. You start with a small number of random samples, say 5, and rank them according to their similarity to your vision of Brad Pitt: 1 for the worst, 5 for the best. You then rate image by image using GPyOpt that works in a latent space of VAE. For the new image, you need to somehow assign a real number that will show how good this image is. The simple idea is to ask a user to compare a new image with previous images (along with their scores). A user then enters score to a current image.
# 
# The proposed scoring has a lot of drawbacks, and you may feel free to come up with new ones: e.g. showing user 9 different images and asking a user which image looks the "best".
# 
# Note that the goal of this task is for you to implement a new algorithm by yourself. You may try different techniques for your task and select one that works the best.

# <b>Task 2:</b> Implement person search using Bayesian optimization. (You can use code from the assignment on Gaussian Processes)
# 
# Note: try varying `acquisition_type` and `acquisition_par` parameters.

# In[ ]:


class FacialComposit:
    def __init__(self, decoder, latent_size):
        self.latent_size = latent_size
        self.latent_placeholder = tf.placeholder(tf.float32, (1, latent_size))
        self.decode = decoder(self.latent_placeholder)
        self.samples = None
        self.images = None
        self.rating = None

    def _get_image(self, latent):
        img = sess.run(self.decode, 
                       feed_dict={self.latent_placeholder: latent[None, :]})[0]
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def _show_images(images, titles):
        assert len(images) == len(titles)
        clear_output()
        plt.figure(figsize=(3*len(images), 3))
        n = len(titles)
        for i in range(n):
            plt.subplot(1, n, i+1)
            plt.imshow(images[i])
            plt.title(str(titles[i]))
            plt.axis('off')
        plt.show()

    @staticmethod
    def _draw_border(image, w=2):
        bordred_image = image.copy()
        bordred_image[:, :w] = [1, 0, 0]
        bordred_image[:, -w:] = [1, 0, 0]
        bordred_image[:w, :] = [1, 0, 0]
        bordred_image[-w:, :] = [1, 0, 0]
        return bordred_image

    def query_initial(self, n_start=10, select_top=6):
        '''
        Creates initial points for Bayesian optimization
        Generate *n_start* random images and asks user to rank them.
        Gives maximum score to the best image and minimum to the worst.
        :param n_start: number of images to rank initialy.
        :param select_top: number of images to keep
        '''
        self.samples = np.zeros([select_top, self.latent_size]) ### YOUR CODE HERE (size: select_top x 64 x 64 x 3)
        self.images = np.zeros([select_top, 64, 64, 3]) ### YOUR CODE HERE (size: select_top x 64 x 64 x 3)
        self.rating = np.zeros([select_top]) ### YOUR CODE HERE (size: select_top)
        
        ### YOUR CODE:
        ### Show user some samples (hint: use self._get_image and input())
        images = []
        titles = []
        scores = []
        codes = []  # list of latent
        
        for i in range(n_start):
            latent_code = np.random.normal(size=[latent_size])  # create latent sample
            codes.append(latent_code)  # add the latent code to the list of latent codes
            
            images.append(self._get_image(latent_code))
            titles.append("image " + str(i+1))
        
        self._show_images(images, titles)
        
        print('Initial images/points for bayesian optimization')
        user_scores = input('Please rate between 1 (worst) to 10 (best), seprate values by -: ')
        for sc in user_scores.split('-'):
            scores.append(int(sc))
        
        # verification for user mistake
        assert(len(scores) == len(images))

        # Sort in descending order and get indices
        indices_desc = sorted(range(len(scores)), key=lambda z: -scores[z])
        
        # keep the first select_top images
        for i, idx in enumerate(indices_desc[:select_top]):
            self.samples[i] = codes[idx]
            self.images[i] = images[idx]
            self.rating[i] = scores[idx]

        # Check that tensor sizes are correct
        np.testing.assert_equal(self.rating.shape, [select_top])
        np.testing.assert_equal(self.images.shape, [select_top, 64, 64, 3])
        np.testing.assert_equal(self.samples.shape, [select_top, self.latent_size])

    def evaluate(self, candidate):
        '''
        Queries candidate vs known image set.
        Adds candidate into images pool.
        :param candidate: latent vector of size 1xlatent_size
        '''
        initial_size = len(self.images)
        
        ### YOUR CODE HERE
        ## Show user an image and ask to assign score to it.
        ## You may want to show some images to user along with their scores
        ## You should also save candidate, corresponding image and rating
        image = self._get_image(candidate[0])
        
        images = list(self.images[:3])# 
        images.append(image)
        
        titles = list(self.rating[:3])# 
        titles.append("candidate")
        
        self._show_images(images, titles)
        
        candidate_rating = int(input("Please rate the candidate image:")) ### YOUR CODE HERE
        
        self.images = np.vstack((self.images, np.array([image])))
        self.rating = np.hstack((self.rating, np.array([candidate_rating])))
        self.samples = np.vstack((self.samples, candidate))
        
        assert len(self.images) == initial_size + 1
        assert len(self.rating) == initial_size + 1
        assert len(self.samples) == initial_size + 1
        return candidate_rating

    def optimize(self, n_iter=10, w=4, acquisition_type='MPI', acquisition_par=0.3):
        if self.samples is None:
            self.query_initial()

        bounds = [{'name': 'z_{0:03d}'.format(i),
                   'type': 'continuous',
                   'domain': (-w, w)} 
                  for i in range(self.latent_size)]
        optimizer = GPyOpt.methods.BayesianOptimization(f=self.evaluate, domain=bounds,
                                                        acquisition_type = acquisition_type,
                                                        acquisition_par = acquisition_par,
                                                        exact_eval=False, # Since we are not sure
                                                        model_type='GP',
                                                        X=self.samples,
                                                        Y=self.rating[:, None],
                                                        maximize=True)
        optimizer.run_optimization(max_iter=n_iter, eps=-1)

    def get_best(self):
        index_best = np.argmax(self.rating)
        return self.images[index_best]

    def draw_best(self, title=''):
        index_best = np.argmax(self.rating)
        image = self.images[index_best]
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        plt.show()


# #### GRADED 2 (3 points):
# Describe your approach below: How do you assign a score to a new image? How do you select reference images to help user assign a new score? What are the limitations of your approach?

# 1- To assign a score to a new image, we rate the image between 1 (the worst) to 10 (the best). The three best-scored images are shown to the user for the fair comparison
# 
# 2- On the first iteration, we show 10 images and select the top 6 images. However on the subsequent iterations we filter the best three images and show it for the fair comparison
# 
# 3- This VAE approach for facial composites has a couple of limistations. First, a latent variable space is used to reconstruct the whole face. So, it makes the approach inaccurate. It would be better to use an approach to score and modify different parts of the face separately. The second limitation is that we may reach loval optima points or the points that are not desired in the direction we imagined for improvement. The third limitation is that since we use subjective socores for the whole face it is diffcult for the user to give fair score to the images.     
# 
# 
# 

# ## Testing your algorithm

# In these sections, we will apply the implemented app to search for different people. Each task will ask you to generate images that will have some property like "dark hair" or "mustache". You will need to run your search algorithm and provide the best discovered image.

# #### Task 3.1: Finding person with darkest hair (3 points)

# In[ ]:


composit = FacialComposit(decoder, 8)
composit.optimize()


# In[ ]:





# In[ ]:


composit.draw_best('Darkest hair')


# #### Task 3.2. Finding person with the widest smile (3 points)

# In[ ]:


composit = FacialComposit(decoder, 8)
composit.optimize()


# In[ ]:


composit.draw_best('Widest smile')


# #### Task 3.3. Finding Daniil Polykovskiy or Alexander Novikov — lecturers of this course (3 points) 

# Note: this task highly depends on the quality of a VAE and a search algorithm. You may need to restart your search algorithm a few times and start with larget initial set.

# In[ ]:


composit = FacialComposit(decoder, 8)
composit.optimize()


# In[ ]:


composit.draw_best('Lecturer')


# #### <small>Don't forget to post resulting image of lecturers on the forum ;)</small>

# #### Task 3.4. Finding specific person (optional, but very cool)

# Now that you have a good sense of what your algorithm can do, here is an optional assignment for you. Think of a famous person and take look at his/her picture for a minute. Then use your app to create an image of the person you thought of. You can post it in the forum <a href="https://www.coursera.org/learn/bayesian-methods-in-machine-learning/discussions/forums/SE06u3rLEeeh0gq4yYKIVA">Final project: guess who!</a>
# 

# In[ ]:


### Your code here


# ### Submission
# You need to share this notebook via a link: click `SHARE` (in the top-right corner) $\rightarrow$ `Get shareable link` and then paste the link into the assignment page.
# 
# **Note** that the reviewers always see the current version of the notebook, so please do not remove or change the contents of the notebook until the review is done.
# 
# ##### If you are working locally (e.g. using Jupyter instead of Colab)
# Please upload your notebook to Colab and share it via the link or upload the notebook to any file sharing service (e.g. Dropbox) and submit the link to the notebook through https://nbviewer.jupyter.org/, so by clicking on the link the reviewers will see it's contents (and will not need to download the file).

# ### Grading criterions

# #### Task 1 (3 points) [samples from VAE]
# * 0 points: No results were provided here or provided images were not generated by VAE
# * 1 point: Provided images poorly resemble faces
# * 2 points: Provided images look like faces, maybe with some artifacts, but all look the same
# * 3 points: Provided images look like faces, maybe with some artifacts, and look different

# #### Task 2 (3 points) [training procedure]
# * 0 points: No result was provided
# * 1 point: Some algorithm was proposed, but it does not use Bayesian optimization
# * 2 points: Algorithm was proposed, but there were no details on some important aspects: how to assign a score to a new image / how to you select a new image / what are the limitations of the approach
# * 3 points: Algorithm was proposed, all questions in the task were answered

# #### Tasks 3.1-3.3 (3 points each) [search for person]
# * 0 points: Nothing was provided
# * 1 point: Resulting image was provided, but some of the required images (evolution & nearest image) are not provided
# * 2 points: All images are provided, but the resulting image does not have requested property
# * 3 points: All images are provided, the resulting image has required features (long hair / wide smile / looks like lecturer)

# ## Passing grade is 60% (9 points)
