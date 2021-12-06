import tensorflow as tf
import keras.backend as K

cos = tf.keras.losses.CosineSimilarity(axis=1)
ones = tf.ones([16, 48, 64, 1], tf.float32)


def accurate_obj_boundaries_loss(depth, output):
    """
    https://arxiv.org/abs/1803.08673
    """
    depth_grad = tf.image.sobel_edges(depth)
    output_grad = tf.image.sobel_edges(output)

    depth_grad_dy = depth_grad[:, :, :, :, 0]
    depth_grad_dx = depth_grad[:, :, :, :, 1]
    output_grad_dy = output_grad[:, :, :, :, 0]
    output_grad_dx = output_grad[:, :, :, :, 1]

    depth_normal = tf.concat([-depth_grad_dx, -depth_grad_dy, ones], axis=-1)
    output_normal = tf.concat([-output_grad_dx, -output_grad_dy, ones], axis=-1)

    # Point-wise depth
    loss_depth = K.mean(K.abs(output - depth), axis=-1)

    # Sobel gradient Loss
    loss_dx = K.mean(K.abs(output_grad_dx - depth_grad_dx), axis=-1)
    loss_dy = K.mean(K.abs(output_grad_dy - depth_grad_dy), axis=-1)

    # Normal Loss
    loss_normal = K.mean(K.abs(1 - cos(output_normal, depth_normal)))

    return K.mean(loss_depth) + K.mean(loss_normal) + K.abs(K.mean(loss_dx) + K.mean(loss_dy))
