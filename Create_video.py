import skvideo.io
import numpy as np
import cv2

video_data = skvideo.io.vreader('videos/201806220910.mp4')


# O = np.zeros(video_data.shape[0:3])

# params = {"-framerate": "15"}
writer = skvideo.io.FFmpegWriter("outputvideo_F_skip.mp4")

Out = []
for i,f in enumerate(video_data):
	if i%8 ==0:
		frame = cv2.resize(f,(400,300),interpolation=cv2.INTER_CUBIC)
	# Out.append(frame)
		writer.writeFrame(frame)
writer.close()

# skvideo.io.vwrite("outputvideo.mp4", np.array(Out))




# outputdata = np.random.random(size=(5, 480, 680, 3)) * 255
# outputdata = outputdata.astype(np.uint8)

# skvideo.io.vwrite("outputvideo.mp4", outputdata)