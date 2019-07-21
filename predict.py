import cv2
import numpy as np

from train import Trainer


def main():
    trainer = Trainer(do_train=False)

    feature_image = cv2.imread("images/rgb.png")
    segmentation_image = trainer.infer(feature_image)
    output_image = (segmentation_image * 255).astype(np.uint8)
    cv2.imwrite("output.png", output_image)


if __name__ == "__main__":
    main()
