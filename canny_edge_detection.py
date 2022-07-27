import numpy as np

class cannyEdgeDetector:
    def __init__(self, img, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=50, highthreshold=100, threshold=130):
        self.img = img
        self.img_gray = None
        self.img_final = None
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        return 

    def _rgb2gray(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray

    def _first_threshold(self, image, t):
        
        #Armazena as dimensões da imagem
        h,w=image.shape
        
        #Cria uma cópia da imagem para não perder a imagem original
        thresh = image.copy()
        
        #Varre a imagem pixel a pixel
        for i in range(h):
            for j in range(w):
                
                #Testa se a intensidade do pixel é menor ou maior que o valor de corte T
                if image[i][j] < t:
                    thresh[i][j] = 0
                else:
                    thresh[i][j] = 255
                    
        #Retorna a imagem limiarizada
        return thresh

    def _gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g

    def _convolution(self, img, kernel, padding=0, strides=1):
        # Cross Correlation
        kernel = np.flipud(np.fliplr(kernel))

        # Gather Shapes of Kernel + img + Padding
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        xImgShape = img.shape[0]
        yImgShape = img.shape[1]

        # Shape of Output Convolution
        xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
        yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
        output = np.zeros((xOutput, yOutput))

        # Apply Equal Padding to All Sides
        if padding != 0:
            imgPadded = np.zeros((img.shape[0] + padding*2, img.shape[1] + padding*2))
            imgPadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = img
            print(imgPadded)
        else:
            imgPadded = img

        # Iterate through img
        for y in range(img.shape[1]):
            # Exit Convolution
            if y > img.shape[1] - yKernShape:
                break
            # Only Convolve if y has gone down by the specified Strides
            if y % strides == 0:
                for x in range(img.shape[0]):
                    # Go to next row once kernel is out of bounds
                    if x > img.shape[0] - xKernShape:
                        break
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            output[x, y] = (kernel * imgPadded[x: x + xKernShape, y: y + yKernShape]).sum()
                    except:
                        break

        return output

    def _sobel_filters(self, img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = self._convolution(img, Kx)
        Iy = self._convolution(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)
    

    def _non_max_suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180


        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0


                except IndexError as e:
                    pass

        return Z

    def _second_threshold(self, img):

        highThreshold = img.max() * self.highThreshold;
        lowThreshold = highThreshold * self.lowThreshold;

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)

    def _hysteresis(self, img):

        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img
    
    def detect(self):
        self.img_gray = self._rgb2gray(self.img)
        self.img_smoothed = self._convolution(self.img_gray, self._gaussian_kernel(self.kernel_size, self.sigma))
        self.img_threshold = self._first_threshold(self.img_smoothed, self.threshold)
        self.gradientMat, self.thetaMat = self._sobel_filters(self.img_threshold)
        self.nonMaxImg = self._non_max_suppression(self.gradientMat, self.thetaMat)
        self.thresholdImg = self._second_threshold(self.nonMaxImg)
        img_final = self._hysteresis(self.thresholdImg)
        self.img_final = img_final

        return self.img_smoothed, self.img_final

