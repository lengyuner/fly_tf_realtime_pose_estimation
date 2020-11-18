
import os
os.path
os.getcwd()


import cv2

print(cv2.__version__)
# videoCapture = cv2.VideoCapture('IMG_2789.MOV')
# C:\Users\dongj\Desktop\courses\AI\Experiment\xihu
name='C:/Users/ps/Desktop/djz/datasets/video_todo/20200423_133926_2/20200423_133926_2.avi'

videoCapture = cv2.VideoCapture(name)

# print(frame.shape)
# Out[10]: (1024, 1280, 3)
fps = 30  # 保存视频的帧率
# size = (1024, 1280)  # 保存视频的大小
size = (1280, 1024)
# videoWriter = cv2.VideoWriter('video4_20201024.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)

OUTPUT_FILE='C:/Users/ps/Desktop/djz/video_20201024.avi'
videoWriter = cv2.VideoWriter(OUTPUT_FILE,
                              cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)

# OUTPUT_FILE='C:/Users/ps/Desktop/djz/video4_20201024.avi'
# (width, height)=size
# videoWriter = cv2.VideoWriter(OUTPUT_FILE,
#                 cv2.VideoWriter_fourcc('I', '4', '2', '0'),
#                 30, # fps
#                 (1280, 1024))
                # (width, height))
i = 0

while True:
    success, frame = videoCapture.read()
    i += 1

    if (i >= 0 and i <= fps * 10):
        print('i = ', i)
        videoWriter.write(frame)
    else:
        print('end')
        break
    # if success:
        # i += 1
        #
        # if (i >= 0 and i <= fps*10):
        #     print('i = ', i)
        #     videoWriter.write(frame)

