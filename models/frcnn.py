import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras.applications.efficientnet import EfficientNetB0,EfficientNetB3,EfficientNetB7
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from cvmodels.custom_layers import ROIPoolingLayer

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

    self.output_deltas = L.Dense(
            units=4 * self.numBbox,
            activation="linear",
            kernel_initializer="glorot_normal",
            name="deltas"
        )
    
    self.flatten = L.Flatten()

    self.output_scores = L.Dense(
            units=num_classes * self.numBbox,
            activation="softmax",
            kernel_initializer="glorot_normal",
            name="scores"
        )
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
  
  def call(self,inputs):
    x = self.backbone(inputs[0])
    x = self.roi_pooling([x,inputs[1]])
    x = self.flatten(x)
    # x = self.fc1(x)
    delta = self.output_deltas(x)
    score = self.output_scores(x)
    return [score,delta]
  
  def build(self):
    inp,roi = self.inputShape,self.roi_shape
    _ = self.call([K.ones((1,inp[0],inp[1],inp[2])),K.ones((1,roi[0],roi[1]))])
    self.built=True