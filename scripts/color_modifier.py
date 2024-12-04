import cv2 as cv
import numpy as np

class ColorModifier:

    def __init__(self):
        pass

    def adjustBrightness(self, image : np.array, value : int = 0) -> np.array:
        """ Adjust Brightness by adding a scaler value to all pixels"""
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        v = cv.add(v, value)
        v = np.clip(v, 0, 255)  # Ensure values stay within [0,255]
        final_hsv = cv.merge((h, s, v))

        return cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    
    def adjustHue(self, image : np.array, value : int = 0) -> np.array:
        """ Adjust Hue by adding a  scaler value to all pixels"""
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        h = cv.add(h, value)
        h = np.clip(h, 0, 255)
        final_hsv = cv.merge((h, s, v))

        return cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    
    def adjustSaturation(self, image : np.array, value : int = 0) -> np.array:
        """ Adjust Saturation by adding a scaler value to all pixels"""
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        s = cv.add(s, value)
        s = np.clip(s, 0, 255)
        final_hsv = cv.merge((h, s, v))

        return cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    
    def adjustContrast(self, image : np.array, factor : float = 0) -> np.array:
        """ Adjust Contrast by multiplying a scaler value to all pixels"""
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        v = np.clip(v * factor, 0, 255).astype(np.uint8)
        final_hsv = cv.merge((h, s, v))

        return cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)

    def applySharpening(self, image : np.array, intensity : int = 0) -> np.array:
        """Apply Sharpening to an image with adjustable intensity"""
        # Define the sharpening filter
        sharpen_filter = np.array([[ 0, -1,  0],
                                    [-1,  5, -1],
                                    [ 0, -1,  0]])

        # Apply the sharpening filter using convolution
        sharpened = cv.filter2D(image, -1, sharpen_filter)

        return self.__applyWithIntensity(image, sharpened, intensity)
    
    def __applyWithIntensity(self, original : np.array, effect : np.array, intensity : int = 0 ) -> np.array:
        """Blend original and effect images based on intensity"""
        intensity = intensity / 100.0  # Normalize intensity to [0, 1]
        return cv.addWeighted(original, 1 - intensity, effect, intensity, 0)
    
    def applySepia(self, image : np.array, intensity : int = 0) -> np.array:
        """Apply Sepia Filter"""
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
        sepia = cv.transform(image, sepia_filter)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)  # Clip to [0,255]

        return self.__applyWithIntensity(image, sepia, intensity)
    
    def applyGaussianBlur(self, image : np.array, kernel : int = 15, intensity : int = 0) -> np.array:
        """Apply Gaussian Blur with controllable intensity"""
        kernel_size = (kernel, kernel)
        blurred = cv.GaussianBlur(image,kernel_size, 0)

        return self.__applyWithIntensity(image, blurred, intensity)

    def applyPencilSketch(self, image : np.array, intensity : int = 0) -> np.array:
        """Convert image to pencil sketch with controllable intensity"""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        inverted = cv.bitwise_not(gray)
        blurred = cv.GaussianBlur(inverted, (21, 21), 0)
        sketch = cv.divide(gray, 255 - blurred, scale=256)
        sketch_bgr = cv.cvtColor(sketch, cv.COLOR_GRAY2BGR)

        return self.__applyWithIntensity(image, sketch_bgr, intensity)

    def detectEdges(self, image : np.array, intensity : int = 0) -> np.array:
        """ Detect edges using filter"""
        edge_filter = np.array([[-1,-1,-1],
                                [-1,8,-1],
                                [-1,-1,-1]])

        edges = cv.filter2D(image, -1, edge_filter)

        return self.__applyWithIntensity(image, edges, intensity)

    def applyWarmth(self, image : np.array, intensity : int = 0):
        """ Apply Warmth using filter"""
        warmth_filter = np.array([[1.2,  0.2,  0.2],
                          [0.2,  1.2,  0.2],
                          [0.2,  0.2,  1.2]])
        
        warmth = cv.filter2D(image, -1, warmth_filter)

        return self.__applyWithIntensity(image, warmth, intensity)


    

if __name__ == "__main__":

    colormodifier = ColorModifier()
    image_path = 'easy-photoshop/images/dog.jpg'
    image = cv.imread(image_path)

    try:
        new_image = colormodifier.applyPencilSketch(image, 100)

        cv.imshow("Original", image)
        cv.imshow("Modified", new_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    except Exception as e:
        print(e)



