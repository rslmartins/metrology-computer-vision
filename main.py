import argparse
from PIL import Image
import numpy as np
from canny_edge_detection import cannyEdgeDetector as ced
from calculate import contoursGeometry as cg

def main():
    img = np.asarray(Image.open(args.step))
 
    detector = ced(img, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
    img_smoothed, img_final = detector.detect()
    Image.fromarray(np.uint8(img_smoothed)).convert('RGB').save('gauss.png')
    Image.fromarray(np.uint8(img_final)).convert('RGB').save('canny.png')

    generator = cg(img, img_final, 62)
    generator.generate_image()

if __name__ == '__main__':
# Load argument
    parser = argparse.ArgumentParser(description="Runner", add_help=True)
    parser.add_argument(
        "--step",
        metavar="stepName",
        help="[./mechanicalpart.png, ./mechanical_part.jpg, ./mechanical-part.jpg]",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main()