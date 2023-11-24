import cv2

# Load an image
img = cv2.imread('guy2_tree114.jpg')

# Apply Gaussian denoising filter with kernel size 5x5 and standard deviation 0
denoised_img = cv2.GaussianBlur(img, (5, 5), 0)

# Display the original and denoised image
cv2.imshow('Original', img)
cv2.imshow('Denoised', denoised_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
