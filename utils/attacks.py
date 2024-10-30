# utils/attacks.py
import tensorflow as tf
import numpy as np

# FGSM Attack
def create_fgsm_adversarial_image(model, image, label, epsilon=0.01):
    image_tensor = tf.convert_to_tensor(image.reshape((1, 32, 32, 3)))
    label_tensor = tf.convert_to_tensor([label])
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label_tensor, prediction)
    gradient = tape.gradient(loss, image_tensor)
    signed_grad = tf.sign(gradient)
    adversarial_image = image_tensor + epsilon * signed_grad
    return tf.clip_by_value(adversarial_image, 0, 1).numpy().squeeze()

# PGD Attack
def create_pgd_adversarial_image(model, image, label, epsilon=0.01, alpha=0.002, num_iter=10):
    image_tensor = tf.convert_to_tensor(image.reshape((1, 32, 32, 3)), dtype=tf.float32)
    adv_image = image_tensor

    for _ in range(num_iter):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(adv_image)
            loss = tf.keras.losses.sparse_categorical_crossentropy([label], prediction)

        gradient = tape.gradient(loss, adv_image)
        adv_image = adv_image + alpha * tf.sign(gradient)
        perturbation = tf.clip_by_value(adv_image - image_tensor, -epsilon, epsilon)
        adv_image = tf.clip_by_value(image_tensor + perturbation, 0, 1)

    return adv_image.numpy().squeeze()


# Grad-CAM Visualization

# def grad_cam(model, image, class_idx, layer_name=None):
#     # Use the specified layer or the last convolutional layer if layer_name is None
#     if layer_name is None:
#         layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]

#     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

#     with tf.GradientTape() as tape:
#         inputs = tf.cast(image, tf.float32)
#         (conv_outputs, predictions) = grad_model(inputs)
#         loss = predictions[:, class_idx]

#     grads = tape.gradient(loss, conv_outputs)
#     if grads is not None:  # Ensure gradients are not None
#         pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#         conv_outputs = conv_outputs[0]
#         heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
#         heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-6)  # Normalize and avoid divide-by-zero
#         return heatmap.numpy()
#     else:
#         print("Gradients were None.")
#         return np.zeros((image.shape[1], image.shape[2]))  # Return a blank heatmap if no gradients were found

def grad_cam(model, image, class_idx, layer_name=None):
    # Check or determine layer name
    if layer_name is None:
        # Find the last convolutional layer if layer_name is not specified
        layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]
    print(f"Using layer: {layer_name}")

    # Create a sub-model that maps the input to the desired layer output and the predictions
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    # Ensure image has the correct shape (add batch dimension if necessary)
    if image.shape != (1, 32, 32, 3):
        image = np.expand_dims(image, axis=0)  # Add batch dimension

    with tf.GradientTape() as tape:
        inputs = tf.cast(image, tf.float32)
        (conv_outputs, predictions) = grad_model(inputs)
        loss = predictions[:, class_idx]  # Target class loss

    # Compute gradients with respect to conv_outputs
    grads = tape.gradient(loss, conv_outputs)
    if grads is not None:  # Check if gradients are not None
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global average pooling
        conv_outputs = conv_outputs[0]  # Remove batch dimension from conv_outputs

        # Create heatmap by combining pooled gradients with conv_outputs
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-6)  # Normalize heatmap
        return heatmap.numpy()  # Return as a numpy array for easy plotting
    else:
        print("Gradients were None.")
        return np.zeros((image.shape[1], image.shape[2]))  # Blank heatmap if no gradients
