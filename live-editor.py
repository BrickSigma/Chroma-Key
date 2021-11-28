import cv2
import numpy as np
import pyvirtualcam

from numba import jit

CUDA_AVAILABLE = False
try:
    from numba import cuda
    if cuda.is_available():
        cuda.detect()
        CUDA_AVAILABLE = True
    else:
        print("No CUDA devices available")
except ImportError:
    print('Could not import CUDA')

from color_range import Analysis  # This is a class built for analysing the background

img = cv2.imread("bg.png")  # Load in the background image path of the video here
HEIGHT, WIDTH = img.shape[:2]
print(f"path.png: WIDTH={WIDTH}, HEIGHT={HEIGHT}")

all_rgb_codes = img.reshape(-1, img.shape[-1])
colors = np.unique(all_rgb_codes, axis=0)
print(f"path.png: {colors.shape[0]} unique colours")

analize = Analysis(colors)  # This creates the analysis object. It holds all the details and function for removing the background
analize.range_format()  # This will create a specialised range of equations used to remove the background (more info in color_range.py file)
equations_xy, equations_zy = analize.get_rules()
print(f"equations_xy: {equations_xy}")
print(f"equations_zy: {equations_zy}")

if CUDA_AVAILABLE:
    equations_xy = cuda.to_device(equations_xy)
    equations_zy = cuda.to_device(equations_zy)

vidcap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Capture live camera feed
success,image = vidcap.read()
height, width, layers = image.shape

bg_present = True
new_bg = cv2.imread("new background.jpg")  # Select a background to show. If background doesn't exist, the frame will not be edited
try:
    y, x = new_bg.shape[:2]
    new_bg = new_bg[y-height:y, x-width:x]
except:
    bg_present = False  # If no background is input, this operator will cancel any edits to the frame.

@jit(nopython=True)
def check_colors(analize,image):
    if bg_present:
        for h in range(height):
            for w in range(width):
                if analize.check_color(image[h, w]):  # Calls the analize class to check if the color is in the editing range
                    image[h, w, 0] = new_bg[h, w, 0]
                    image[h, w, 1] = new_bg[h, w, 1]
                    image[h, w, 2] = new_bg[h, w, 2]


if CUDA_AVAILABLE:
    # CUDA kernel (executes on device)
    @cuda.jit()
    def cuda_check_color_kernel(A,B,equations_xy,equations_zy):
        i, j = cuda.grid(2)
        if i < A.shape[0] and j < A.shape[1] and bg_present:
            color = (A[i,j,0],A[i,j,1],A[i,j,2])
            if (((equations_xy[0,0]*color[0])+equations_xy[0,1] >= color[1]) and
               ((equations_xy[1,0]*color[0])+equations_xy[1,1] <= color[1]) and
               ((equations_xy[2,0]*color[0])+equations_xy[2,1] >= color[1]) and
               ((equations_xy[3,0]*color[0])+equations_xy[3,1] <= color[1]) and
               ((equations_zy[0,0]*color[2])+equations_zy[0,1] >= color[1]) and
               ((equations_zy[1,0]*color[2])+equations_zy[1,1] <= color[1]) and
               ((equations_zy[2,0]*color[2])+equations_zy[2,1] >= color[1]) and
               ((equations_zy[3,0]*color[2])+equations_zy[3,1] <= color[1])):
                B[i,j,0] = new_bg[i,j,0]
                B[i,j,1] = new_bg[i,j,0]
                B[i,j,2] = new_bg[i,j,0]
            else:
                B[i,j,0] = A[i,j,0]
                B[i,j,1] = A[i,j,1]
                B[i,j,2] = A[i,j,2]

    BLOCKDIM_X = 8
    BLOCKDIM_Y = 4

    def cuda_check_colors(A,eq_xy,eq_zy):
        blockdim = BLOCKDIM_Y, BLOCKDIM_X
        griddim = (height+BLOCKDIM_Y-1) // BLOCKDIM_Y, (width+BLOCKDIM_X-1) // BLOCKDIM_X
        stream = cuda.stream()
        with stream.auto_synchronize():
            d_Result = cuda.device_array_like(A, stream) # create d_Result array (on device), with same size as A
            d_A = cuda.to_device(A, stream) # copy A to d_A (on device)
            cuda_check_color_kernel[griddim, blockdim, stream](d_A,d_Result,eq_xy,eq_zy) # execute the kernel
            d_Result.copy_to_host(ary=A,stream=stream) # copy from d_Result (on device) back to A (on host)

# Main loop that changes each frame and outputs it.
with pyvirtualcam.Camera(width=width, height=height, fps=60) as cam:
    while success:
        
        if CUDA_AVAILABLE:
            cuda_check_colors(image, equations_xy, equations_zy)
        else:
            check_colors(analize, image)

        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert color mode from BGR to RGB
        cam.send(img)  # Output the new frame to the virtual camera
        success,image = vidcap.read()  # Read the next frame
        cam.sleep_until_next_frame()
