# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DELF model implementation based on the following paper.

  Large-Scale Image Retrieval with Attentive Deep Local Features
  https://arxiv.org/abs/1612.06321

Please refer to the README.md file for detailed explanations on using the DELF
model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import resnet_v1

slim = tf.contrib.slim

_SUPPORTED_TARGET_LAYER = ['resnet_v1_50/block3', 'resnet_v1_50/block4']

# The variable scope for the attention portion of the model.
_ATTENTION_VARIABLE_SCOPE = 'attention_block'

# The attention_type determines whether the attention based feature aggregation
# is performed on the L2-normalized feature map or on the default feature map
# where L2-normalization is not applied. Note that in both cases, attention
# functions are built on the un-normalized feature map. This is only relevant
# for the training stage.
# Currently supported options are as follows:
# * use_l2_normalized_feature:
#   The option use_l2_normalized_feature first applies L2-normalization on the
#   feature map and then applies attention based feature aggregation. This
#   option is used for the DELF+FT+Att model in the paper.
# * use_default_input_feature:
#   The option use_default_input_feature aggregates unnormalized feature map
#   directly.
_SUPPORTED_ATTENTION_TYPES = [
    'use_l2_normalized_feature', 'use_default_input_feature'
]

# Supported types of non-lineary for the attention score function.
_SUPPORTED_ATTENTION_NONLINEARITY = ['softplus']


class DelfV1(object):
  """Creates a DELF model.

  Args:
    target_layer_type: The name of target CNN architecture and its layer.

  Raises:
    ValueError: If an unknown target_layer_type is provided.
  """

  def __init__(self, target_layer_type=_SUPPORTED_TARGET_LAYER[0]):
    tf.logging.info('Creating model %s ', target_layer_type)

    self._target_layer_type = target_layer_type
    if self._target_layer_type not in _SUPPORTED_TARGET_LAYER:
      raise ValueError('Unknown model type.')

  @property
  def target_layer_type(self):
    return self._target_layer_type

  def _PerformAttention(self,
                        attention_feature_map,
                        feature_map,
                        attention_nonlinear,
                        kernel=1):
    """Helper function to construct the attention part of the model.

    Computes attention score map and aggregates the input feature map based on
    the attention score map.

    Args:
      attention_feature_map: Potentially normalized feature map that will
        be aggregated with attention score map.
            
      feature_map: Unnormalized feature map that will be used to compute
        attention score map.
      attention_nonlinear: Type of non-linearity that will be applied to
        attention value.
      kernel: Convolutional kernel to use in attention layers (eg: 1, [3, 3]).

    Returns:
      attention_feat: Aggregated feature vector.
      attention_prob: Attention score map after the non-linearity.
      attention_score: Attention score map before the non-linearity.

    Raises:
      ValueError: If unknown attention non-linearity type is provided.
    """
    with tf.variable_scope('attention', values=[attention_feature_map, feature_map]):
      with tf.variable_scope('compute', values=[feature_map]):
        
        # attention Network        
        # kernel 논문에서는 1x1이나 여기서는 찾을길이 없음.ㅠ 그러나, WidthxHeight크기를 유지하면서 channel -> 1로 가는게 목적이기때문에,
        #    1x1이 맞을듯~
        # 1. 첫번째 network는 relu형태의 512(채널) outpout를 같는 layers        
        activation_fn_conv1 = tf.nn.relu
        feature_map_conv1   = slim.conv2d(feature_map, 512, kernel, rate=1, activation_fn=activation_fn_conv1, scope='conv1')
        # 2. 두 번째 network는 1(채널) outpout를 같는 layers (뒤에 activated f = softplus)
        attention_score     = slim.conv2d( feature_map_conv1, 1, kernel, rate=1, activation_fn=None, normalizer_fn=None, scope='conv2')

      # Set activation of conv2 layer of attention model.
      with tf.variable_scope('merge', values=[attention_feature_map, attention_score]):
        if attention_nonlinear not in _SUPPORTED_ATTENTION_NONLINEARITY:
          raise ValueError('Unknown attention non-linearity.')
          
        if attention_nonlinear == 'softplus':          
          with tf.variable_scope('softplus_attention', values=[attention_feature_map, attention_score]):
            
            # 3. attention score 계산 
            #   softplus activation : 두번째 cnn 결과를 입력으로 받는다.            
            attention_prob = tf.nn.softplus(attention_score)
            
            # 4. attention feature map : attention score를 conv feature map에 적용
            #  attention feature map = attention_feature_map x attention_prob
            #      attention_feature_map = 3차원, WxHxC(=512)
            #      attention_prob        = 2차원, WxHxC(=1)              
            # 주의) attention_feature_map = 입력으로 들어올 때, 오리지널 conv feature map 형태이던가 이를 l2 norm하던가의 형태로 들어오게 됨
            #       다시 말해서, conv feature map일뿐 이름에 혹하지 말자  
            attention_feat = tf.reduce_mean(tf.multiply(attention_feature_map, attention_prob), [1, 2])
            
        # tf.expand_dim을 두번하는 이유는 ??? > [batch, 1, 1, channels] 이 크기를 유지한는데
        #   요 때의 channels는  width x height 의 크기?? channel크기??
        #     > tf.reduce_mean(A, [1,2])에서 채널은 유지하면서 withxheight의 평균(average pooling)에 더 점수를.. = channel크기
        #     > paper에서 n개의 d-dim의 가진 feature sum은 n번 해야하니..d-dim의 크기를 가져야하는듯~ d-dim = channel
        attention_feat = tf.expand_dims(tf.expand_dims(attention_feat, 1), 2)
        #
        # 참고로 이후에) def _GetAttentionModel() 함수에서, fcn이 num_classes를 prediction (softmax 연산과 비슷하게)
        #   logits  = slim.conv2d(attention_feat, num_classes, [1, 1],  activation_fn=None, normalizer_fn=None, scope='logits')
        #   logits  = tf.squeeze(logits, [1, 2], name='spatial_squeeze') # [batch, num_classes]
        #   실제적으로 _GetAttentionModel 안불리는것으로 보아, training 단계에서만 사용하는듯보임.
        
        # attention_prob: Attention score after the non-linearity. run softplus
        # attention_score: Attention score before the non-linearity. not softplus
    return attention_feat, attention_prob, attention_score

  def _GetAttentionSubnetwork(
      self,
      feature_map,
      end_points,
      attention_nonlinear=_SUPPORTED_ATTENTION_NONLINEARITY[0],
      attention_type=_SUPPORTED_ATTENTION_TYPES[0],
      kernel=1,
      reuse=False):
    """Constructs the part of the model performing attention.

    Args:
      feature_map: A tensor of size [batch, height, width, channels]. Usually it
        corresponds to the output feature map of a fully-convolutional network.
      end_points: Set of activations of the network constructed so far.
      attention_nonlinear: Type of non-linearity on top of the attention
        function.
      attention_type: Type of the attention structure.
      kernel: Convolutional kernel to use in attention layers (eg, [3, 3]).
      reuse: Whether or not the layer and its variables should be reused.

    Returns:
      prelogits: A tensor of size [batch, 1, 1, channels].
      attention_prob: Attention score after the non-linearity.
      attention_score: Attention score before the non-linearity.
      end_points: Updated set of activations, for external use.
    Raises:
      ValueError: If unknown attention_type is provided.
    """
    with tf.variable_scope(
        _ATTENTION_VARIABLE_SCOPE,
        values=[feature_map, end_points],
        reuse=reuse):
      if attention_type not in _SUPPORTED_ATTENTION_TYPES:
        raise ValueError('Unknown attention_type.')
      if attention_type == 'use_l2_normalized_feature':
        attention_feature_map = tf.nn.l2_normalize(
            feature_map, 3, name='l2_normalize')
      elif attention_type == 'use_default_input_feature':
        attention_feature_map = feature_map
      end_points['attention_feature_map'] = attention_feature_map

      attention_outputs = self._PerformAttention(
          attention_feature_map, feature_map, attention_nonlinear, kernel)
      prelogits, attention_prob, attention_score = attention_outputs
      end_points['prelogits']       = prelogits
      end_points['attention_prob']  = attention_prob
      end_points['attention_score'] = attention_score
    return prelogits, attention_prob, attention_score, end_points

  def GetResnet50Subnetwork(self,
                            images,
                            is_training=False,
                            global_pool=False,
                            reuse=None):
    """Constructs resnet_v1_50 part of the DELF model.

    Args:
      images: A tensor of size [batch, height, width, channels].
      is_training: Whether or not the model is in training mode.
      global_pool: If True, perform global average pooling after feature
        extraction. This may be useful for DELF's descriptor fine-tuning stage.
      reuse: Whether or not the layer and its variables should be reused.

    Returns:
      net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
        If global_pool is True, height_out = width_out = 1.
      end_points: A set of activations for external use.
    """
    block = resnet_v1.resnet_v1_block
    blocks = [
        block('block1', base_depth=64, num_units=3, stride=2),
        block('block2', base_depth=128, num_units=4, stride=2),
        block('block3', base_depth=256, num_units=6, stride=2),
    ]
    if self._target_layer_type == 'resnet_v1_50/block4':
      blocks.append(block('block4', base_depth=512, num_units=3, stride=1))
    net, end_points = resnet_v1.resnet_v1(
        images,
        blocks,
        is_training=is_training,
        global_pool=global_pool,
        reuse=reuse,
        scope='resnet_v1_50')
    return net, end_points

  # 실제 feature extraction에서는 main 함수인듯~
  def GetAttentionPrelogit(
      self,
      images,
      weight_decay=0.0001,
      attention_nonlinear=_SUPPORTED_ATTENTION_NONLINEARITY[0],
      attention_type=_SUPPORTED_ATTENTION_TYPES[0],
      kernel=1,
      training_resnet=False,
      training_attention=False,
      reuse=False,
      use_batch_norm=True):
    """Constructs attention model on resnet_v1_50.

    Args:
      images: A tensor of size [batch, height, width, channels].
      weight_decay: The parameters for weight_decay regularizer.
      attention_nonlinear: Type of non-linearity on top of the attention
        function.
      attention_type: Type of the attention structure.
      kernel: Convolutional kernel to use in attention layers (eg, [3, 3]).
      training_resnet: Whether or not the Resnet blocks from the model are in
        training mode.
      training_attention: Whether or not the attention part of the model is
        in training mode.
      reuse: Whether or not the layer and its variables should be reused.
      use_batch_norm: Whether or not to use batch normalization.

    Returns:
      prelogits: A tensor of size [batch, 1, 1, channels].
      attention_prob: Attention score after the non-linearity.
      attention_score: Attention score before the non-linearity.
      feature_map: Features extracted from the model, which are not
        l2-normalized.
      end_points: Set of activations for external use.
    """
    # Construct Resnet50 features.
    with slim.arg_scope(
        resnet_v1.resnet_arg_scope(use_batch_norm=use_batch_norm)):
      _, end_points = self.GetResnet50Subnetwork(
          images, is_training=training_resnet, reuse=reuse)

    feature_map = end_points[self._target_layer_type]

    # Construct attention subnetwork on top of features.
    with slim.arg_scope(
        resnet_v1.resnet_arg_scope(
            weight_decay=weight_decay, use_batch_norm=use_batch_norm)):
      with slim.arg_scope([slim.batch_norm], is_training=training_attention):
        (prelogits, attention_prob, attention_score,
         end_points) = self._GetAttentionSubnetwork(
             feature_map,
             end_points,
             attention_nonlinear=attention_nonlinear,
             attention_type=attention_type,
             kernel=kernel,
             reuse=reuse)

    return prelogits, attention_prob, attention_score, feature_map, end_points

  def _GetAttentionModel(
      self,
      images,
      num_classes,
      weight_decay=0.0001,
      attention_nonlinear=_SUPPORTED_ATTENTION_NONLINEARITY[0],
      attention_type=_SUPPORTED_ATTENTION_TYPES[0],
      kernel=1,
      training_resnet=False,
      training_attention=False,
      reuse=False):
    """Constructs attention model on resnet_v1_50.

    Args:
      images: A tensor of size [batch, height, width, channels]
      num_classes: The number of output classes.
      weight_decay: The parameters for weight_decay regularizer.
      attention_nonlinear: Type of non-linearity on top of the attention
        function.
      attention_type: Type of the attention structure.
      kernel: Convolutional kernel to use in attention layers (eg, [3, 3]).
      training_resnet: Whether or not the Resnet blocks from the model are in
        training mode.
      training_attention: Whether or not the attention part of the model is in
        training mode.
      reuse: Whether or not the layer and its variables should be reused.

    Returns:
      logits: A tensor of size [batch, num_classes].
      attention_prob: Attention score after the non-linearity.
      attention_score: Attention score before the non-linearity.
      feature_map: Features extracted from the model, which are not
        l2-normalized.
    """

    attention_feat, attention_prob, attention_score, feature_map, _ = (
        self.GetAttentionPrelogit(
            images,
            weight_decay,
            attention_nonlinear=attention_nonlinear,
            attention_type=attention_type,
            kernel=kernel,
            training_resnet=training_resnet,
            training_attention=training_attention,
            reuse=reuse))
    
    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay, batch_norm_scale=True)):
      with slim.arg_scope([slim.batch_norm], is_training=training_attention):
        with tf.variable_scope(_ATTENTION_VARIABLE_SCOPE, values=[attention_feat], reuse=reuse):
                    
          # 사실상 fcn > num_classes predict > softmax역할 아님 뒤에서 하는가??
          # _PerformAttention의 결과를 가지고 
          # attention_feat = [batch, 1, 1, channel]
          logits = slim.conv2d(attention_feat, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
          logits = tf.squeeze(logits, [1, 2], name='spatial_squeeze') # [batch, num_classes]
          
    return logits, attention_prob, attention_score, feature_map

  def AttentionModel(self,
                     images,
                     num_classes,
                     weight_decay=0.0001,
                     attention_nonlinear=_SUPPORTED_ATTENTION_NONLINEARITY[0],
                     attention_type=_SUPPORTED_ATTENTION_TYPES[0],
                     kernel=1,
                     training_resnet=False,
                     training_attention=False,
                     reuse=False):
    """Constructs attention based classification model for training.

    Args:
      images: A tensor of size [batch, height, width, channels]
      num_classes: The number of output classes.
      weight_decay: The parameters for weight_decay regularizer.
      attention_nonlinear: Type of non-linearity on top of the attention
        function.
      attention_type: Type of the attention structure.
      kernel: Convolutional kernel to use in attention layers (eg, [3, 3]).
      training_resnet: Whether or not the Resnet blocks from the model are in
        training mode.
      training_attention: Whether or not the model is in training mode. Note
        that this function only supports training the attention part of the
        model, ie, the feature extraction layers are not trained.
      reuse: Whether or not the layer and its variables should be reused.

    Returns:
      logit: A tensor of size [batch, num_classes]
      attention: Attention score after the non-linearity.
      feature_map: Features extracted from the model, which are not
        l2-normalized.

    Raises:
      ValueError: If unknown target_layer_type is provided.
    """
    if 'resnet_v1_50' in self._target_layer_type:
      net_outputs = self._GetAttentionModel(
          images,
          num_classes,
          weight_decay,
          attention_nonlinear=attention_nonlinear,
          attention_type=attention_type,
          kernel=kernel,
          training_resnet=training_resnet,
          training_attention=training_attention,
          reuse=reuse)
      logits, attention, _, feature_map = net_outputs
    else:
      raise ValueError('Unknown target_layer_type.')
    return logits, attention, feature_map
