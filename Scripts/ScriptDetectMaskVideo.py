import sys
from utilities import *
import cv2
import json
import os
from tqdm import tqdm
import tensorflow.keras as keras
import moviepy.editor as moviepy

if len(sys.argv) != 6:
    print("Please, insert 5 arguments:\n1) Path to the video, including its format.\n2) Path to the tensorflow model.\n3) Path to the JSON file containing anchors and labels.\n4) Boolean: True or False in order to use GPU or not.\n5) Number from 0 to 1 indicating the threshold below which the detection is not considered.")
    sys.exit()

if sys.argv[4] == "True":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    import tensorflow as tf
    print("Marking GPU as device with ID 0. If your GPU is not in that ID, please change it manually.")
elif sys.argv[4] == "False":
    print("We are not using GPU...")
    import tensorflow as tf
else:
    print("Write literally 'True' or 'False' if you want to use GPU or not.")
    sys.exit()

if float(sys.argv[5]) <= 0 or float(sys.argv[5]) > 1:
    print("Please, indicate a value for the threshold bigger than 0 and 1 as a maximum.")
    sys.exit()

def add_audio():
    videocap = cv2.VideoCapture(sys.argv[1])
    video = moviepy.VideoFileClip(sys.argv[1])
    audio = video.audio
    audio.write_audiofile("audio.mp3")
    my_clip = moviepy.VideoFileClip("output.avi")
    final_clip = my_clip.set_audio(moviepy.AudioFileClip("audio.mp3"))
    final_clip.write_videofile("output2.mp4", fps = videocap.get(cv2.CAP_PROP_FPS))

def main():
    input_w, input_h = 416, 416

    model = keras.models.load_model(sys.argv[2])

    file = open(sys.argv[3])
    json_dictionary = json.load(file)
    file.close()

    labels = json_dictionary["labels"]
    anchors = json_dictionary["anchors"]

    class_threshold = float(sys.argv[5])

    vidcap = cv2.VideoCapture(sys.argv[1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    success, frame = vidcap.read()
    out = cv2.VideoWriter('output.avi',fourcc, vidcap.get(cv2.CAP_PROP_FPS), (frame.shape[1],frame.shape[0]))
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in tqdm(range(length-1)):
        ret, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image, image_w, image_h = load_image_stream(frame, (input_w, input_h))
        predicciones = model.predict(image)

        boxes = list()
        for i in range(len(predicciones)):
            boxes += decode_netout(predicciones[i][0], anchors[i], class_threshold, input_h, input_w)

        correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
        do_nms(boxes, 0.5)
        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

        for i in range(len(v_boxes)):
            y1, x1, y2, x2 = v_boxes[i].ymin, v_boxes[i].xmin, v_boxes[i].ymax, v_boxes[i].xmax
            if v_labels[i] == "with_mask":
                frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
                frame = cv2.putText(frame, "mask"+" "+str(np.round(v_scores[i], 2)), (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif v_labels[i] == "without_mask":
                frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 1)
                frame = cv2.putText(frame, "no_mask"+" "+str(np.round(v_scores[i], 2)), (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (255,165,0), 1)
                frame = cv2.putText(frame, v_labels[i]+" "+str(np.round(v_scores[i], 2)), (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)

    vidcap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
     main()
     add_audio()
     os.remove("audio.mp3")
     os.remove("output.avi")