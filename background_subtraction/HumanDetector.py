import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure, io

import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, image = cap.read()

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), visualize=True, multichannel=True)

    """     
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image') """

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    """     
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show() """

    cv2.imshow('Video', hog_image_rescaled)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
