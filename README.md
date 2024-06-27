# Coffee Leaf Classification

This project aims to develop a machine learning model to classify coffee leaves based on their visual characteristics. Correct classification of leaves can help identify diseases, pests, and improve coffee production.

## Objectives

- Develop a machine learning model to classify coffee leaves.
- Analyze different visual characteristics of the leaves.
- Evaluate the model's accuracy in identifying different classes of leaves.

## Technologies Used

- Python
- TensorFlow/Keras
- OpenCV
- Pandas
- Matplotlib

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/fellipeptc/models_coffee_leaf.git
    cd coffee-leaf-classification
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

## Example usage

1. Prepare your dataset of coffee leaf images and organize it into folders corresponding to classes.
2. Run the training script:
    ```bash
    python train_model.py --dataset path/to/dataset --epochs 50
    ```
3. To classify new images, use the classification script:
    ```bash
    python classify.py --image path/to/image
    ```
    
## Contact

If you have any questions, please contact:

- Name: Fellipe Prates
- Email: fellipeptc@hotmail.com
