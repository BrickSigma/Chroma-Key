# <ins>Chroma-Key</ins>
This is my own version of chroma key software using python and OpenCV.  

## Different options
1. Video editor
2. Live stream editor

## Video Editor
The video editor program is in the `video-editor.py` file.  
Here are the steps to using it:  
1. Take a photograph of your background,
2. Save it and enter it's path into line `18` of the file (where it has `bg.png` entered),
3. Copy the path of the background you want to replace the previous background with into line `47` (where `new background.jpg` is),
4. Record your video,
5. Load the video into the file in line `37` (where `video.mp4` is),
6. Enter the name of the edited file in line `38` (where `test.mp4` is),
7. Run the program and wait until it has completed.
  
Your video has now been edited and the background has been removed.  
(__Note:__ The final video will not contain any audio. I plan on fixing that soon.)

## Live Stream Editor
The live stream editor is located in the `live-editor.py` file.  
To use it you will first need to install [OBS](https://obsproject.com/) which is a virtual camera. You may also have to install the virtual camera plugin for it, which you can download [here](https://obsproject.com/forum/resources/obs-virtualcam.539/).  
  
  Once installed, follow these steps:
1. Repeat steps 1 to 3 from before, with the current background path entered into line `20` of the file and the new background path entered into line `43`.
2. Run the program. You're camera should automatically start up.
3. Open up the app you want to use the program in and change it's camera to `OBS Virtual Camera` (or a different virtual camera if you have one).
  
  You can now use the chroma key while in meetings or livestreams.  

## Conditions for best results
1. Use a `blue` or `green` background; this is a usual standard and prevents parts of your body from being edited out.
2. Make sure to wear contrasting colors to the background.
3. Extra lighting of the background can help to remove shadows.  

## How my chroma key works
The chroma key works by analysing the background of the video and producing 8 line equations. The background is loaded into the program and all of the unique color values in it are extracted into a list.  
  
  This list is then passed over to a class called `Analysis`. Within this class there are 3 main functions: `split_coord`, `calc_offset`, and `check_color`.  
    
When the list of colors is passed through the `__init__` function, it is split into three arrays containg the individual color channels of the background (B,G,R). These arrays are called `xs`, `ys`, and `zs` with respect to the color channels.  
  
The first function, `split_coord` is the main function that does all the calculations. It takes two of the color channels and is called twice in the entire program; the first time the `xs` and `ys` (blue and green colors) arrays are passed in, and the second time the `zs` and `ys` (red and green colors) arrays are passed in.  
The function treats these as a pair of coordinates on a 2 dimensional plane. It then gets the maximum domain and range of these points, producing two coordinates. These coordinates are used to create the equation of the `central line` in the format `y = mx + c`. It then splits the points into two lists, one list contains the points above the central line and the other contains the points bellow the central line.  
These lists are then used to find the furthest point perpendicular to the central line. Once these points are determined, the program can construc the first four line equations that surround all the points in a rectangle.  
  
  The second function in the class is the `calc_offset` and by it's name it is used to calculate an offset value to add to the y-intercept of the four equations generated. It works by getting the standard deviation of all the values stored in the three arrays (`xs`, `ys`, `zs`) and then halfs it. This value allows the program to account for the changing light levels in the background and it is added to the equations generated in the `split_coord` function.  
      
After the `split_coord` function has been called twice, a total of 8 line equation each with the format `y = mx + c` have been generated. These equations are then used in the final function, `check_color` which will check if the current color value in the frame is within the range of the line equations. If it is, the function will return `True` to the main program and edit the frame's pixel color.  
  
  And that's how my chroma key works.  

## <ins>Notes</ins>:
1. The video editor currently removes the audio from the file, but I'm working on a fix for that.  
2. I plan on making a simple GUI interface for the programs so that it will be more user friendly.
3. I do plan on explaining how my program works in more depth with images soon.
  
## <ins>To-Do:</ins>
1. Restore audio to edited videos,
2. Create a siple GUI for the programs,
3. Run more tests to fix any small bugs.
