###================================BASE MODEL SWIN BEGIN -ALTERNATE======================================
import tensorflow as tf

from transformers import TFSwinModel#, AutoImageProcessor


def getSwinTBackBone():
    base_model2 = TFSwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224",  output_hidden_states = True)
    input1 = tf.keras.Input((3,224,224))  
    output2 = base_model2(input1)
    o1 = output2.reshaped_hidden_states[-1]
    o2 = output2.reshaped_hidden_states[-3]
    o1 = tf.keras.layers.Permute((2,3,1)) (o1)   
    o2 = tf.keras.layers.Permute((2,3,1)) (o2)
    
    return tf.keras.models.Model(base_model2.input, [o1, o2])

def getSwinLBackBone():
    base_model2 = TFSwinModel.from_pretrained("microsoft/swin-large-patch4-window12-384-in22k",  output_hidden_states = True)
    input1 = tf.keras.Input((3,384,384))  
    output2 = base_model2(input1)
    o1 = output2.reshaped_hidden_states[-1]
    o2 = output2.reshaped_hidden_states[-3]
    o1 = tf.keras.layers.Permute((2,3,1)) (o1)   
    o2 = tf.keras.layers.Permute((2,3,1)) (o2)
    
    return tf.keras.models.Model(base_model2.input, [o1, o2])

'''
from transformers import AutoImageProcessor
from models import make_top_model_v2
import numpy as np
image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-large-patch4-window12-384-in22k")
X_train_aug = image_processor(np.ones((5,384,384,3)), return_tensors="tf")['pixel_values']



###=============================FINAL TEST SCRIPT BEGIN===================================###
base_model2 = getSwinLBackBone()

H, W = 384, 384
top_model = make_top_model_v2((int(H/32),int(W/32),1536), (int(H/16),int(W/16),768))

model = tf.keras.models.Model(base_model2.input, top_model(base_model2.output))
###=============================FINAL TEST SCRIPT END=====================================###


###=============================DEBUG SCRIPT BEGIN===================================###
base_model = TFSwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224",  output_hidden_states = True)
output1 = base_model(tf.ones((5,3,224,224)))

base_model2 = TFSwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224",  output_hidden_states = True)
input1 = tf.keras.Input((3,224,224))  
output2 = base_model2(input1)

print("\noutput dimensions:")

print("\nusing first method \n")

for i in range(len(output1.hidden_states)):
    print(output1.hidden_states[i].shape)

print("\nusing second method \n")
print("hidden states:\n")
for i in range(len(output2.hidden_states)):
    print(output2.hidden_states[i].shape)
print("reshaped hidden states: \n")    
for i in range(len(output2.reshaped_hidden_states)):
    print(output2.reshaped_hidden_states[i].shape)
    
o1 = output2.reshaped_hidden_states[-1]
o2 = output2.reshaped_hidden_states[-3]

# h1, w1, d1 = int(np.sqrt(o1.shape[1])), int(np.sqrt(o1.shape[1])), int(o1.shape[-1])
# o1 = tf.keras.layers.Reshape((h1,w1,d1)) (o1)   
o1 = tf.keras.layers.Permute((2,3,1)) (o1)   
# h2, w2, d2 = int(np.sqrt(o2.shape[1])), int(np.sqrt(o2.shape[1])), int(o2.shape[-1])
# o2 = tf.keras.layers.Reshape((h2,w2,d2)) (o2)
o2 = tf.keras.layers.Permute((2,3,1)) (o2)

print("\no1 layer output shape")  
print(o1.shape)
print("\no2 layer output shape")  
print(o2.shape)

###================================TOP MODEL BEGIN======================================
H, W = 224, 224
top_model = make_top_model_v2((int(H/32),int(W/32),768), (int(H/16),int(W/16),384))

###================================TOP MODEL END========================================

top_model([o1,o2])

model = tf.keras.models.Model(base_model2.input, top_model([o1, o2]))

print("\nmodel.input\n ",model.input)
print("\nmodel.output\n ",model.output)

test = model(tf.ones((5,3,224,224)))
print("\ntest[0].shape\n ", test[0].shape)
print("test[1].shape\n ", test[1].shape)
test2 = model(tf.ones((5,3,224,224)), training = True)
print("\ntest2[0].shape\n ", test2[0].shape)
print("test2[1].shape\n ", test2[1].shape)

###=============================DEBUG SCRIPT END===================================###
'''