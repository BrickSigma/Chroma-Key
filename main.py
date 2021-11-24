import cv2
import numpy as np

from color_range import Analysis  # This is a class built for analysing the background

img = cv2.imread("bg.png")
HEIGHT, WIDTH = img.shape[:2]

colors = set({})  # I'm using sets as they're much faster to loop through

#  Add all of the different colors present in the background to the set
for h in range(HEIGHT):
    for w in range(WIDTH):
        if tuple(img[h, w]) in colors:
            continue
        else:
            colors.add((int(img[h, w][0]), int(img[h, w][1]), int(img[h, w][2])))

colors = list(colors)  # Convert the set to a list so that it can be used by the class
analize = Analysis(colors)  # This creates the analysis object. It holds all the details and function for removing the background
analize.range_format()  # This will create a specialised range of equations used to remove the background (more info in color_range.py file)


vidcap = cv2.VideoCapture('video.mp4')  # Load in the video you want edited
video_name = 'test.mp4'  # This is the new edited video's name
fps = vidcap.get(cv2.CAP_PROP_FPS)
success,image = vidcap.read()  # Reads the first frame
height, width, layers = image.shape
video = cv2.VideoWriter(video_name, 0, fps, (width,height))  # Set up the video writer 


# Main loop that goes through every frame in the video and edits them
count = 0
while success:
    
    # Loop through the frame's pixels, this part is slow.
    for h in range(height):
        for w in range(width):
            if analize.check_color(image[h, w]):  # This calls one of the functions in the Analize class that chacks whether the color is in in the set range or not. This is one of the slow parts
                image[h, w] = np.array((0, 0, 255))  # Change the color to red
    
    video.write(image)  # Add the new frame to the video
    
    success,image = vidcap.read()  # Read the next frame
    print('Read a new frame: ', success)
    count += 1

cv2.destroyAllWindows()
video.release()  # Release the video
