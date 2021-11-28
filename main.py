import cv2
import numpy as np
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
analize.calc_offset()
analize.range_format()  # This will create a specialised range of equations used to remove the background (more info in color_range.py file)
equations_xy, equations_zy = analize.get_rules()
print(f"equations_xy: {equations_xy}")
print(f"equations_zy: {equations_zy}")

if CUDA_AVAILABLE:
    equations_xy = cuda.to_device(equations_xy)
    equations_zy = cuda.to_device(equations_zy)

vidcap = cv2.VideoCapture("video.mp4")  # Load in the video you want edited
video_name = 'test.mp4'  # This is the new edited video's name
fps = vidcap.get(cv2.CAP_PROP_FPS)
success,image = vidcap.read()  # Reads the first frame
height, width, layers = image.shape
print(f"video.mp4: Width={width}, Height={height}")

video = cv2.VideoWriter(video_name, 0, 30, (width,height))  # Set up the video writer 

@jit(nopython=True)
def check_colors(analize,image):
    for h in range(height):
        for w in range(width):
            if analize.check_color(image[h, w]):  # This calls one of the functions in the Analize class that chacks whether the color is in in the set range or not. This is one of the slow parts
                # Change the color to red
                image[h, w, 0] = 0
                image[h, w, 1] = 0
                image[h, w, 2] = 255


if CUDA_AVAILABLE:
    # CUDA kernel (executes on device)
    @cuda.jit()
    def cuda_check_color_kernel(A,B,equations_xy,equations_zy):
        i, j = cuda.grid(2)
        if i < A.shape[0] and j < A.shape[1]:
            color = (A[i,j,0],A[i,j,1],A[i,j,2])
            if (((equations_xy[0,0]*color[0])+equations_xy[0,1] >= color[1]) and
               ((equations_xy[1,0]*color[0])+equations_xy[1,1] <= color[1]) and
               ((equations_xy[2,0]*color[0])+equations_xy[2,1] >= color[1]) and
               ((equations_xy[3,0]*color[0])+equations_xy[3,1] <= color[1]) and
               ((equations_zy[0,0]*color[2])+equations_zy[0,1] >= color[1]) and
               ((equations_zy[1,0]*color[2])+equations_zy[1,1] <= color[1]) and
               ((equations_zy[2,0]*color[2])+equations_zy[2,1] >= color[1]) and
               ((equations_zy[3,0]*color[2])+equations_zy[3,1] <= color[1])):
                B[i,j,0] = 0
                B[i,j,1] = 0
                B[i,j,2] = 255
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

# Main loop that goes through every frame in the video and edits them
while success:
    
    if CUDA_AVAILABLE:
        cuda_check_colors(image, equations_xy, equations_zy)
    else:
        check_colors(analize, image)

    video.write(image)  # Add the new frame to the video

    success,image = vidcap.read()  # Read the next frame

vidcap.release()
video.release()  # Release the video
cv2.destroyAllWindows()
