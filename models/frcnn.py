import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras.applications.efficientnet import EfficientNetB0,EfficientNetB3,EfficientNetB7
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from cvmodels.custom_layers import ROIPoolingLayer

def classificationLoss(max_bbox_per_image):
  def getitems_by_indices(self,values, indices):
    return tf.map_fn(
        lambda x: tf.gather(x[0], x[1]), (values, indices), dtype=values.dtype
    )

  def loss(y_true,y_pred):
    _, indices = tf.nn.top_k(tf.argmax(y_pred,axis=-1),max_bbox_per_image)
    y_pred = getitems_by_indices(y_pred,indices)
    y_true = getitems_by_indices(y_true,indices)
    cce = tf.keras.losses.CategoricalCrossentropy()
    l = tf.reduce_mean(tf.map_fn(lambda x: cce(x[0],x[1]),(y_true,y_pred)))
    return l
  return loss

# def regressionLoss()

class FastRCNN(tf.keras.Model):
  def __init__(self,
               inputShape: list,
               max_rois: int,
               backbone: str,
               num_classes: int,
               max_bbox_per_image: int
               ):
    assert backbone in ["InceptionV3","VGG16","EfficientNetB0","EfficientNetB3","EfficientNetB7"],"{} backbone not supported".format(backbone)
    super(FastRCNN,self).__init__()
    self.inputShape,self.roi_shape = inputShape
    
    self.roi_pooling = ROIPoolingLayer(7,7)
    # self.fc1 = L.Dense(
    #             units=256,
    #             activation="relu",
    #             name="fc1"
    #            )
    self.numBbox = max_bbox_per_image

    self.output_deltas = L.TimeDistributed(L.Dense(
            				   units=4 * num_classes,
            				   activation="linear",
            				   kernel_initializer="glorot_normal"
        				  ), name="deltas")
    
    self.flatten = L.TimeDistributed(L.Flatten())

    self.output_scores = L.TimeDistributed(L.Dense(
            				   units=num_classes,
            				   activation="softmax",
            				   kernel_initializer="glorot_normal"
        				   ),name="scores")
    if self.roi_shape[0]>max_rois:
      self.roi_shape = (max_rois,self.roi_shape[1])
    if backbone=="InceptionV3":
      self.backbone = InceptionV3(include_top=False,weights="imagenet",input_shape=self.inputShape)
    elif backbone =="VGG16":
      self.backbone = VGG16(include_top=False,weights="imagenet",input_shape=self.inputShape)
    elif backbone=="EfficientNetB0":
      self.backbone = EfficientNetB0(include_top=False,weights="imagenet",input_shape=self.inputShape)
    elif backbone=="EfficientNetB3":
      self.backbone = EfficientNetB3(include_top=False,weights="imagenet",input_shape=self.inputShape)
    elif backbone=="EfficientNetB7":
      self.backbone = EfficientNetB7(include_top=False,weights="imagenet",input_shape=self.inputShape)
    
    self.backbone.trainable= False
  
  def getitems_by_indices(self,values, indices):
    return tf.map_fn(
        lambda x: tf.gather(x[0], x[1]), (values, indices), dtype=values.dtype
    )
  
  def call(self,inputs):
    if inputs[1].shape[1]>self.roi_shape[1]:
      inputs[1] = inputs[1][:self.roi_shape[1]]

    x = self.backbone(inputs[0])
    x = self.roi_pooling([x,inputs[1]])
    x = self.flatten(x)
    # x = self.fc1(x)
    deltas = self.output_deltas(x)
    scores = self.output_scores(x)
    return [scores,deltas]
  
  def build(self):
    inp,roi = self.inputShape,self.roi_shape
    _ = self.call([K.ones((1,inp[0],inp[1],inp[2])),K.ones((1,roi[0],roi[1]))])
    self.built=True