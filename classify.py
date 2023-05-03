import argparse
import numpy as np
from PIL import Image
from transformers import pipeline

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--model', type=str)
    return parser.parse_args()

def main():
    args = get_args()

    image = Image.open(args.image)
    classifier = pipeline("image-classification", model=args.model, device=0)
    results = classifier(image)

    for result in results:
        print(f"Label: {result['label']} / Score: {result['score']}")
    
if __name__ == '__main__':
    main()