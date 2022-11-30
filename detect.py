# %%
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold

scale = 4

# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
	classes = f.read().rstrip('\n').split('\n')
 
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3-tiny.cfg"
modelWeights = "yolov3-tiny.weights"
 
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# %%
# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

# %%
# Draw the predicted bounding box
def drawPred(frame, classId, conf, left, top, right, bottom):
	# Draw a bounding box.
	cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
	 
	label = '%.2f' % conf
		 
	# Get the label for the class name and its confidence
	if classes:
		assert(classId < len(classes))
		label = '%s: %s' % (classes[classId], label)
 
	#Display the label at the top of the bounding box
	font = cv2.FONT_HERSHEY_SIMPLEX
	labelSize, baseLine = cv2.getTextSize(label, font, 0.5, 1)
	top = max(top, labelSize[1])
	#cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
	cv2.putText(frame, label, (left, top - round(1.5*labelSize[1])), font, 1, (255,255,0))

# %%
# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
	frameHeight = frame.shape[0]
	frameWidth = frame.shape[1]
 
	# Scan through all the bounding boxes output from the network and keep only the
	# ones with high confidence scores. Assign the box's class label as the class with the highest score.
	classIds = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = scores[classId]
			if confidence > confThreshold:
				center_x = int(detection[0] * frameWidth)
				center_y = int(detection[1] * frameHeight)
				width = int(detection[2] * frameWidth)
				height = int(detection[3] * frameHeight)
				left = int(center_x - width / 2)
				top = int(center_y - height / 2)
				classIds.append(classId)
				confidences.append(float(confidence))
				boxes.append([left, top, width, height])
 
	# Perform non maximum suppression to eliminate redundant overlapping boxes with
	# lower confidences.
	indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
	for i in indices:
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)

# %%
while cap.isOpened():
	flags, img = cap.read()
	img = cv2.flip(img, 1) # if your camera reverses your image
	img_resized = cv2.resize(img, (int(img.shape[1]*(4/3)), int(img.shape[1]*(3/4)))) # convert to 3/4
	

	# must be multiple of 32
	big_img = cv2.resize(img_resized, (512,384))
	small_img = cv2.resize(img_resized, (int(big_img.shape[1]/scale), int(big_img.shape[0]/scale)))

	blob = cv2.dnn.blobFromImage(small_img, 1/255,(small_img.shape[1],small_img.shape[0]),[0,0,0],crop=False)
	net.setInput(blob)

	outs = net.forward(getOutputsNames(net))
	postprocess(img, outs)

	t, _ = net.getPerfProfile()
	label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
	cv2.putText(img, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


	cv2.imshow("image", img)
	k=cv2.waitKey(1) & 0XFF
	if k== 27:
		break

cap.release()
cv2.destroyAllWindows() 


