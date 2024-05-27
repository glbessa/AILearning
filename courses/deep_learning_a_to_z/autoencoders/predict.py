import tensorflow as tf
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

def encode_image(model, image):
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape((1, np.prod(image.shape)))
    encoded_image = model.encoder(image)
    return encoded_image

def decode_image(model, encoded_image):
    image = model.decoder(encoded_image) * 255.0
    image = image.numpy().reshape((28, 28))
    return image

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["encode", "decode"])
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=False)
    args = parser.parse_args()

    # Load model
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path {args.model_path} does not exist")
    
    model = load_model(args.model_path)

    # Load input
    if not os.path.exists(args.input_path):
        raise ValueError(f"Input path {args.input_path} does not exist")
    
    if args.mode == "encode":
        image = Image.open(args.input_path)
        encoded_image = encode_image(model, image)
        if args.output_path:
            np.save(args.output_path, encoded_image)
        else:
            np.save("encoded_image.npy", encoded_image)
    else:
        encoded_image = np.load(args.input_path)
        decoded_image = decode_image(model, encoded_image)
        decoded_image = Image.fromarray(decoded_image)
        decoded_image = decoded_image.convert("L")
        if args.output_path:
            decoded_image.save(args.output_path)
        else:
            decoded_image.save("decoded_image.png")
    

