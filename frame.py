import cv2
vidcap = cv2.VideoCapture('/home/abid/Downloads/deepspeech.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("/home/abid/Downloads/id/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1