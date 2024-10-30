# streamlit_app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
import os
from utils.attacks import create_fgsm_adversarial_image, create_pgd_adversarial_image, grad_cam

tf.config.run_functions_eagerly(True)

# Define CIFAR-100 class names
class_names = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "computer_keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger",
    "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

# Load the pre-trained model
model = load_model("models/cifar100_model.h5")

# Initialize the model by running a dummy input through it
dummy_input = np.zeros((1, 32, 32, 3))  # Adjust the shape based on your model's expected input
_ = model.predict(dummy_input)

# Streamlit App Layout
st.image("Header.png", use_column_width=True)
#st.title("Adversarial Attacks and Defense")
st.write("""
    This interactive app lets you explore how specific techniques, like "adversarial attacks," can change how a trained deep-learning model interprets images. 
    In our case, from the CIFAR-100 dataset (a collection of 100 different categories of pictures). 

    You can upload a photo and apply one of these techniques—Fast Gradient Sign Method (FGSM) or Projected Gradient Descent (PGD)—to see how they affect the model's predictions. Doing this lets you observe how even small changes in an image can make the model misinterpret what it "sees."
""")

with st.expander("What are Adversarial Attacks?"):
    st.write("""
        Adversarial attacks are techniques to subtly alter input data (e.g., images) so that a machine learning model misclassifies it.
        These perturbations are often small enough that they’re imperceptible to humans but can significantly impact the model’s performance.
    """)
    st.write("In this demo, we’ll explore two types of attacks:")
    st.markdown("""
    - **Fast Gradient Sign Method (FGSM)**: A single-step attack that perturbs the image in the direction of the model’s gradient.
    - **Projected Gradient Descent (PGD)**: A multi-step attack that iteratively applies FGSM, making it more powerful.
    """)

# Divider line
st.markdown("---")

# About the CIFAR-100 Dataset
st.header("Understanding the CIFAR-100 Dataset")
st.write("""
    CIFAR-100 is a dataset containing 60,000 color images across 100 classes. Each class contains 600 images of 32x32 pixels.
    The dataset is widely used for training and evaluating machine learning models, especially in image classification.
    Here are some popular classes you can explore:
""")
st.write(", ".join(class_names[:20]) + "...")
st.write("For the best results, try uploading an image that resembles one of these classes!")


# Divider line
st.markdown("---")

# Instructions
st.header("How to Use This App")
st.markdown("""
1. **Upload an Image**: Upload an image that will be resized to match CIFAR-100 dimensions.
2. **Select the Correct Label**: Choose the correct label for the uploaded image from the CIFAR-100 classes.
3. **Choose Attack Type**: Select either FGSM or PGD to apply to the image.
4. **Adjust Attack Parameters**:
   - **Epsilon**: Strength of the perturbation. Higher values increase the attack's effectiveness but may make the changes more visible.
   - **Alpha** and **Iterations** (for PGD): Fine-tune the PGD attack with step size and iteration count.
5. **Generate Adversarial Image**: View the model's predictions on the attacked image and Grad-CAM visualizations.
""")

with st.expander("Detailed Explanations & Reasoning"):
    st.write("""

        1. **Upload an Image**: Select an image file (JPEG or PNG) to upload, 
        which will be automatically resized to CIFAR-100’s standard 32x32 pixel dimensions for compatibility with the model.

        2. **Select the Correct Label**: Adversarial attacks, like Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), 
        work by introducing small perturbations to the input image in a way that causes the model to misclassify it.
        To generate a meaningful adversarial example, the algorithm needs to know the true label of the image. 
        This label guides the attack to make the model's prediction diverge from the correct classification, increasing the chance of successful misclassification.

        3. **Choose Attack Type**: Select an adversarial attack method—either FGSM (a single-step attack) or PGD (a multi-step, iterative attack)—to modify the image.

        4. **Adjust Attack Parameters**:

        Epsilon: Set the strength of the perturbation; higher values create stronger attacks but may make alterations in the image more noticeable.

        Alpha and Iterations (for PGD): For PGD, adjust the step size (alpha) and the number of iterations to control the power and subtlety of the attack.

        5. **Generate Adversarial Image**: Run the attack and observe the model’s predictions and Grad-CAM visualizations, 
        which highlight the areas the model considers important before and after the attack.
    """)

# Divider line
st.markdown("---")

# Choose between upload or sample image
st.header("Choose an Image")
image_option = st.radio("Select an option to choose an image:", ("Upload an Image", "Select a Sample Image"))

# Image selection logic
if image_option == "Upload an Image":
    uploaded_file = st.file_uploader("Upload an Image (JPEG/PNG)", type=["jpg", "png"])
    if uploaded_file:
        # Load and preprocess the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((32, 32))  # Resize to CIFAR-100 dimensions
        image_np = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
else:
    # List all sample images
    sample_images_dir = "sample_images"
    sample_images = [img for img in os.listdir(sample_images_dir) if img.endswith((".jpg", ".png"))]
    selected_sample_image = st.selectbox("Select a sample image:", sample_images)

    # Load and preprocess the selected sample image
    if selected_sample_image:
        image_path = os.path.join(sample_images_dir, selected_sample_image)
        image = Image.open(image_path).convert("RGB")
        image = image.resize((32, 32))  # Resize to CIFAR-100 dimensions
        image_np = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

# Display Original Image and Label Selection if image is chosen
if 'image' in locals():
    st.image(image, caption="Selected Image (pre-processed to 32x32 CIFAR-100 dimensions)", width=300)

    # Label Selection
    st.subheader("1. Select the Correct Label")
    label = st.selectbox(
        "Select the correct label for the chosen image:",
        range(100),
        format_func=lambda x: class_names[x]
    )

    # Divider line
    st.markdown("---")

    # Attack Type and Parameters
    st.subheader("2. Choose Attack Type and Parameters")
    attack_type = st.radio("Choose an attack type:", ["FGSM", "PGD"])

    # Set attack parameters
    epsilon = st.slider("Epsilon (Strength of Attack)", 0.0, 0.1, 0.01,
                        help="Higher epsilon means stronger perturbations.")
    if attack_type == "PGD":
        alpha = st.slider("Alpha (Step Size for PGD)", 0.001, 0.01, 0.002, help="Step size for each PGD iteration.")
        num_iter = st.slider("Iterations for PGD", 1, 20, 10, help="Number of iterations for the PGD attack.")

    # Divider line
    st.markdown("---")

    # Generate adversarial image
    st.subheader("3. Generate Adversarial Image and View Results")
    if st.button("Generate Adversarial Image"):
        # Generate adversarial image
        if attack_type == "FGSM":
            adv_image = create_fgsm_adversarial_image(model, image_np, label, epsilon)
        else:
            adv_image = create_pgd_adversarial_image(model, image_np, label, epsilon, alpha, num_iter)

        # Side-by-side display of original and adversarial images
        st.write("### Original and Adversarial Images")
        col1, col2 = st.columns(2)
        col1.image(image, caption="Original Image", use_column_width=True)
        col2.image(adv_image, caption=f"{attack_type} Adversarial Image", use_column_width=True)

        # Display predictions
        original_pred = np.argmax(model.predict(image_np.reshape(1, 32, 32, 3)), axis=1)[0]
        adv_pred = np.argmax(model.predict(adv_image.reshape(1, 32, 32, 3)), axis=1)[0]

        st.write(f"**Original Prediction**: {class_names[original_pred]}")
        st.write(f"**Adversarial Prediction**: {class_names[adv_pred]}")

        # Explanation for CNN Vulnerability to Attacks
        # st.header("Why are CNNs Vulnerable to Adversarial Attacks?")
        with st.expander("Was the Attack successful in spoofing our model? "
                         "Learn why models are vulnerable to Adversarial Attacks?"):
            st.write("""
                 Deep learning models learn patterns in data, but adversarial attacks exploit their limitations by introducing subtle changes. 
                 These changes, though invisible to humans, disrupt the model's learned patterns, causing misclassification. 
                 This vulnerability arises from the model's reliance on specific features in the data rather than holistic understanding, making it sensitive to small, targeted perturbations.
             """)

        # Grad-CAM Visualizations
        st.write("### Grad-CAM Visualization")
        st.markdown("""
            The Grad-CAM visualization highlights areas in the image that the model considers important for its prediction.
            Compare the Grad-CAM heatmaps of the original and adversarial images to see how the attack changes the model’s focus.
        """)
        cam_original = grad_cam(model, image_np.reshape(1, 32, 32, 3), original_pred)
        cam_adv = grad_cam(model, adv_image.reshape(1, 32, 32, 3), adv_pred)

        # Side-by-side display of Grad-CAM images
        col1, col2 = st.columns(2)
        col1.image(cam_original, caption="Grad-CAM on Original Image", use_column_width=True)
        col2.image(cam_adv, caption="Grad-CAM on Adversarial Image", use_column_width=True)

        # Divider line
        st.markdown("---")

        # Prompt to try incorrect labels
        st.write("### Curious about the Effect of Incorrect Labels?")
        st.write("""
            Try selecting a label that doesn't match the image and run the adversarial attack to observe the result! 

            **Hint:** If you don't provide the true label, the model is less likely to be fooled by the attack. This is because adversarial attacks target specific weaknesses related to the true class, pushing the model to misclassify it. 
            Without knowing the true label, the attack might not find an effective way to disrupt the model's prediction.
        """)

        # Explanation about correct vs. incorrect label
        with st.expander("Why did the attack work (or fail)?"):
            st.write("""
                  When creating an adversarial example, the attack targets specific weaknesses related to the true label, 
                  using subtle modifications to nudge the model toward a misclassification. If you provided the correct label, 
                  the attack has a clear direction and is more likely to succeed. 

                  However, when the wrong label is provided, the attack lacks this information and may not effectively disrupt the model’s pattern recognition.
                  This is why choosing the correct label is essential for a successful attack. Try experimenting with different labels to see how it affects the attack's effectiveness!
              """)

        # Divider line
        st.markdown("---")

        # Defensive Techniques Section
        st.write("### How Can We Defend Against Adversarial Attacks?")

        with st.expander("Understanding Defense Mechanisms"):
            st.write("""
                Adversarial attacks are challenging to defend against, but researchers have developed several strategies to make models more resilient. Here are some of the main approaches:

                - **Adversarial Training**: This technique involves augmenting the training dataset with adversarial examples, helping the model learn to recognize and resist adversarial perturbations. Adversarial training has proven to be one of the most effective defenses, though it can be computationally expensive.

                - **Input Preprocessing**: Simple transformations, like blurring, resizing, or adding noise, can sometimes disrupt the adversarial perturbations, reducing the attack's impact. By slightly modifying the input before feeding it to the model, these techniques can mitigate some attacks, though they are less effective against stronger, adaptive attacks.

                - **Model Ensembling**: Using multiple models with different architectures or training methods to make predictions collectively can reduce the impact of adversarial attacks. This ensemble approach makes it harder for an adversarial example crafted for one model to successfully fool all models in the ensemble, increasing robustness.

                - **Detection Mechanisms**: Detection methods identify and reject adversarial samples before they reach the model. Techniques include statistical analysis, uncertainty estimation, and anomaly detection. These mechanisms attempt to flag suspicious inputs, but false positives and adaptive attacks can still pose challenges.

                - **Gradient Masking**: Some defense techniques obfuscate the model's gradient, making it difficult for attackers to calculate precise perturbations. However, this is a "security through obscurity" approach and can be bypassed by more advanced attacks that don’t rely on gradient information directly.

                - **Randomization Techniques**: Randomizing certain aspects of the model, such as neuron activations or input preprocessing steps, can make it harder for attackers to reliably produce adversarial examples. This approach, however, may introduce noise in predictions and reduce model performance slightly.

                Researchers are continuously exploring these and hybrid methods, as each defense technique has its trade-offs in terms of effectiveness, computational cost, and impact on model accuracy.
            """)

        # Divider line
        st.markdown("---")


        # Novel Attack Methods Section
        st.write("### Exploring Novel Attack Methods")

        with st.expander("Learn About Advanced Adversarial Attack Techniques"):
            st.write("""
                As defenses against adversarial attacks evolve, so do the attack methods themselves. Here are some advanced attack techniques that have emerged:

                - **Carlini & Wagner (C&W) Attack**: This is a powerful, optimization-based attack that generates minimal perturbations, making it challenging to detect. The C&W attack is highly effective against models that use defensive distillation or gradient masking.

                - **DeepFool Attack**: This attack method iteratively perturbs the image, aiming to find the smallest perturbation needed to change the model’s decision. DeepFool is particularly effective in revealing the minimal vulnerability of the model.

                - **Momentum Iterative Method (MIM)**: An enhancement of PGD, this attack uses momentum to escape local minima in the loss landscape, making it more successful against defenses that rely on iterative attacks. MIM can produce stronger adversarial examples by overcoming obstacles in the optimization path.

                - **AutoAttack (AA)**: This is a suite of four different attacks (including PGD and C&W) designed to systematically test model robustness. AutoAttack has become a popular benchmark for evaluating model robustness, as it combines multiple attack methods into a single framework.

                - **Universal Adversarial Perturbations**: Unlike typical adversarial attacks, which target a specific input, this technique generates a single perturbation that can fool the model on a wide range of images. These universal perturbations expose the vulnerability of models across multiple inputs.

                - **Adversarial Patches**: This method involves creating a physical patch that can be placed in real-world scenes to fool computer vision systems. Adversarial patches have practical applications in fooling facial recognition and object detection systems.

                These novel attack methods demonstrate that as defenses improve, attack strategies also advance, creating an ongoing arms race in the field of adversarial machine learning. Understanding these techniques helps researchers develop stronger, more resilient models capable of withstanding increasingly sophisticated attacks.
            """)

        # Divider line
        st.markdown("---")

        st.write("### Additional Resources")

        st.write("""
            For more detailed documentation and project resources, please check out the following links:

            - 📄 **Project Documentation**: [Notion Notebook](https://tinyurl.com/AdversarialAttack-Defense)
            - 💻 **GitHub Repository**: [GitHub Project](https://github.com/Sarvesh-Vishnu/Adversarial-Attack-Defense)

            These resources contain in-depth explanations of the project's design, implementation, and future improvements. Feel free to explore and contribute!
        """)





