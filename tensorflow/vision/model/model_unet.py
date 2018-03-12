"""Define the model."""

import tensorflow as tf

# Taken from https://github.com/kkweon/UNet-in-Tensorflow/blob/master/train.py
def conv_conv_pool(input_, n_filters, training, name, pool=True, activation=tf.nn.relu, reg=0.1):
    """{Conv -> BN -> RELU}x2 -> {Pool, optional}
    Args:
        input_ (4-D Tensor): (batch_size, H, W, C)
        n_filters (list): number of filters [int, int]
        training (1-D Tensor): Boolean Tensor
        name (str): name postfix
        pool (bool): If True, MaxPool2D
        activation: Activaion functions
    Returns:
        net: output of the Convolution operations
        pool (optional): output of the max pooling operations
    """
    net = input_

    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            net = tf.layers.conv2d(
                net,
                F, (3, 3),
                activation=None,
                padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
                name="conv{}".format(i + 1))
            net = tf.layers.batch_normalization(net, training=training, name="bn{}".format(i + 1))
            net = activation(net, name="relu{}_{}".format(name, i + 1))

        if pool is False:
            return net

        pool = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name="pool_{}".format(name))
        return net, pool

def upconv_concat(inputA, input_B, n_filter, name, reg=0.1):
    """Upsample `inputA` and concat with `input_B`
    Args:
        input_A (4-D Tensor): (N, H, W, C)
        input_B (4-D Tensor): (N, 2*H, 2*H, C2)
        name (str): name of the concat operation
    Returns:
        output (4-D Tensor): (N, 2*H, 2*W, C + C2)
    """
    up_conv = tf.layers.conv2d_transpose(
        inputA,
        filters=n_filter,
        kernel_size=2,
        strides=2,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(reg),
        name="upsample_{}".format(name))

    return tf.concat([up_conv, input_B], axis=-1, name="concat_{}".format(name))

def compute_iou(logits, labels):
    s_logits = tf.nn.softmax(logits)
    s_labels = tf.cast(labels, tf.float32)
    inter = tf.reduce_sum(tf.multiply(s_logits,s_labels))
    union=tf.reduce_sum(tf.subtract(tf.add(s_logits,s_labels),tf.multiply(s_logits,s_labels)))
    iou = tf.divide(inter,union)
    return iou


def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    images = inputs['images']
    labels = inputs['labels']

    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, params.image_channels]
    assert labels.get_shape().as_list() == [None, params.image_size, params.image_size, params.image_classes]

    # Define the number of channels of each convolution
    # For each down layer, we do: 3x3 conv (same) -> BN -> relu -> 3x3 conv (same) -> BN -> relu -> 2x2 maxpool
    nc = params.num_channels
    bn_momentum = params.bn_momentum
    reg = params.regularization
    conv1, pool1 = conv_conv_pool(images, [nc, nc], is_training, name=1, reg=reg)
    conv2, pool2 = conv_conv_pool(pool1, [nc*2, nc*2], is_training, name=2, reg=reg)
    conv3, pool3 = conv_conv_pool(pool2, [nc*4, nc*4], is_training, name=3, reg=reg)
    conv4, pool4 = conv_conv_pool(pool3, [nc*8, nc*8], is_training, name=4, reg=reg)

    # For bottom layer, we do: 3x3 conv (same) -> BN -> relu -> 3x3 conv (same) -> BN -> relu
    conv5 = conv_conv_pool(pool4, [nc*16, nc*16], is_training, name=5, pool=False, reg=reg)
    
    # For each up block, we do: upconv(prev_layer) -> concat(sibling_layer) -> (3x3 conv (same) -> BN -> relu) x2
    up6 = upconv_concat(conv5, conv4, nc*8, name=6, reg=reg)
    conv6 = conv_conv_pool(up6, [nc*8, nc*8], is_training, name=6, pool=False, reg=reg)

    up7 = upconv_concat(conv6, conv3, nc*4, name=7, reg=reg)
    conv7 = conv_conv_pool(up7, [nc*4, nc*4], is_training, name=7, pool=False, reg=reg)

    up8 = upconv_concat(conv7, conv2, nc*2, name=8, reg=reg)
    conv8 = conv_conv_pool(up8, [nc*2, nc*2], is_training, name=8, pool=False, reg=reg)

    up9 = upconv_concat(conv8, conv1, nc, name=9, reg=reg)
    conv9 = conv_conv_pool(up9, [nc, nc], is_training, name=9, pool=False, reg=reg)
    
    # print("Conv" + str(9) + ": " + str(conv9.get_shape().as_list()))
    assert conv9.get_shape().as_list() == [None, params.image_size, params.image_size, nc]

    logits = tf.layers.conv2d(
        conv9,
        params.image_classes, (1,1),
        activation = tf.nn.relu,
        padding = 'same',
        name = 'final')
    # print("Logits: " + str(logits.get_shape().as_list()))
    return logits


def model_unet(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels_class = tf.argmax(labels, 3)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(is_training, inputs, params)
        predictions = tf.argmax(logits, 3)

    # Define loss and accuracy
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_class, predictions), tf.float32))
    iou = compute_iou(logits, labels)

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels_class, predictions=predictions),
            'loss': tf.metrics.mean(loss),
            'iou': tf.metrics.mean_iou(labels=labels_class, predictions=predictions, num_classes=params.image_classes)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('iou', iou)
    tf.summary.image('train_image', tf.reverse(inputs['images'][:,:,:,:3], [-1]))
    tf.summary.image('train_label', tf.cast(tf.expand_dims(labels_class,-1), tf.float32))
    tf.summary.image('pred_label', tf.cast(tf.expand_dims(predictions,-1), tf.float32))

    # #TODO: if mode == 'eval': ?
    # # Add incorrectly labeled images
    # mask = tf.not_equal(labels, predictions)

    # # Add a different summary to know how they were misclassified
    # for label in range(0, params.num_labels):
    #     mask_label = tf.logical_and(mask, tf.equal(predictions, label))
    #     incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
    #     tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['predictions'] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['iou'] = iou
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
