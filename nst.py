#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from IPython import display


# In[63]:


tf.config.experimental_run_functions_eagerly(True)


# In[22]:


def vgg_layers(layer_names):
  vgg = tf.keras.applications.VGG19(include_top=False,weights='imagenet')
  vgg.trainable = False
  outputs = [vgg.get_layer(name).output for name in layer_names]
  model = tf.keras.Model([vgg.input],outputs)
  return model

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd',input_tensor,input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2],tf.float32)
  return result/num_locations


# In[23]:


class StyleContentModel(tf.keras.models.Model):
  def __init__(self,style_layers,content_layers):
    super(StyleContentModel,self).__init__()
    self.vgg = vgg_layers(style_layers+content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False


  def call(self,inputs):
    inputs = inputs*255
    preprocessed_inputs = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_inputs)
    style_outputs,content_outputs = (outputs[:self.num_style_layers],outputs[self.num_style_layers:])
    style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
    content_dict = {content_name:value for content_name,value in zip(self.content_layers,content_outputs)}
    style_dict = {style_name:value for style_name,value in zip(self.style_layers,style_outputs)}
    return {'content':content_dict,'style':style_dict}


# In[59]:


class NeuralStyleTransfer():
    def __init__(self):
        self.content_layers = ['block5_conv2',
                 'block4_conv3']
        self.style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
        self.style_weight = 1e-2
        self.content_weight = 1e4
        self.total_variation_weight = 30
        self.optimizer = tf.optimizers.Adam(learning_rate=0.02,beta_1=0.99,epsilon=1e-1)

    def load_img(self,path_to_img):
        """Function to preprocess image, and restricts maximum dimension image to 512px"""
        max_dim = 512
        img = tf.io.read_file(path_to_img) #reading the file
        img = tf.image.decode_image(img,channels=3) # decoding the file as image file
        img = tf.image.convert_image_dtype(img,dtype=tf.float32) #converting px dtypes to float32
        shape = tf.cast(tf.shape(img)[:-1],tf.float32)
        long_dim = max(shape)
        scale = max_dim/long_dim
        new_shape = tf.cast(shape*scale,dtype=tf.int32)
        img = tf.image.resize(img,new_shape)
        img = img[tf.newaxis,:]#adding a dimension to img tensor, for batch size.
        return img

    def vgg_layers(self,layer_names):
        vgg = tf.keras.applications.VGG19(include_top=False,weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input],outputs)
        return model
    def gram_matrix(self,input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd',input_tensor,input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2],tf.float32)
        return result/num_locations


    def tensor_to_image(self,tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)

    def clip_0_1(self,image):
        return tf.clip_by_value(image,clip_value_min=0.0,clip_value_max=1.0)
    def high_pass_x_y(self,image):
        x_var = image[:,:,1:,:]-image[:,:,:-1,:]
        y_var = image[:,1:,:,:]-image[:,:-1,:,:]
        return x_var,y_var

    def total_variation_loss(self,image):
        x_deltas,y_deltas = self.high_pass_x_y(image)
        return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

    def style_content_loss(self,outputs,style_targets,content_targets):

        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
        style_loss *= self.style_weight/self.num_style_layers
        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
        content_loss *= self.content_weight/self.num_content_layers
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(self,image,style_targets,content_targets):
        extractor = StyleContentModel(self.style_layers,self.content_layers)
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = self.style_content_loss(outputs,style_targets,content_targets)
            loss += self.total_variation_loss(image)*self.total_variation_weight
        # print(outputs,loss)
        grad = tape.gradient(loss,image)
        self.optimizer.apply_gradients([(grad,image)])
        image.assign(self.clip_0_1(image))


    def perform_nst(self,img_path,style_image_path,epochs=10):
        img = self.load_img(img_path)
        style_image = self.load_img(style_image_path)
        extractor = StyleContentModel(self.style_layers,self.content_layers)
        result = extractor(tf.constant(img))
        result = extractor(tf.constant(img))
        style_targets = extractor(style_image)['style']
        content_targets = extractor(img)['content']
        image = tf.Variable(img)
        start = time.time()

        steps_per_epoch = 10
        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step +=1
                self.train_step(image,style_targets,content_targets)
                print('.',end="")
#             display.clear_output(wait=True)

            print('epoch:{},   Train Step: {}'.format(n,step))

        end = time.time()
        print('Total Time: {:.1f}'.format(end-start))
        return self.tensor_to_image(image)










# In[60]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
