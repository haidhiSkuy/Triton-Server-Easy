import math
import numpy as np
import cv2
import tritonclient.http as httpclient 


def preprocess_image(image_path):
    """
    Preprocesses an image to match the input requirements of the model specified in config.pbtxt.

    Args:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Preprocessed image ready for inference.
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    # Resize the image to 224x224
    image = cv2.resize(image, (224, 224))
    
    # Convert the image to float32
    image = image.astype(np.float32)
    
    # Normalize the image (assuming normalization is required)
    # Assuming typical normalization for pretrained models:
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image / 255.0 - mean) / std
    
    # Transpose the image to convert HWC -> CHW format (required for FORMAT_NCHW)
    image = np.transpose(image, (2, 0, 1))
    
    # Add a batch dimension to make it (1, 3, 224, 224)
    # image = np.expand_dims(image, axis=0)
    
    return image


if __name__ == "__main__":
    # Setting up client
    client = httpclient.InferenceServerClient(url="localhost:8000")
    image_input = preprocess_image("tes.jpg")

    detection_input = httpclient.InferInput(
        "data_0", image_input.shape, datatype="FP32"
    )
    detection_input.set_data_from_numpy(image_input, binary_data=True) 

    detection_response = client.infer(
        model_name="densenet_onnx", inputs=[detection_input]
    )

    result = detection_response.as_numpy("fc6_1")
    print(result)