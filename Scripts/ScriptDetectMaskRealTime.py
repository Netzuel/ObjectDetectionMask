import sys
from utilities import *
import cv2
import os
import json

if len(sys.argv) != 5:
    print("Please, insert 4 arguments:\n1) Path to the tflite model.\n2) Path to the JSON file containing anchors and labels.\n3) Boolean: True or False in order to use GPU or not.\n4) Number from 0 to 1 indicating the threshold below which the detection is not considered.")
    sys.exit()

if sys.argv[3] == "True":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    import tensorflow as tf
    print("Marking GPU as device with ID 0. If your GPU is not in that ID, please change it manually.")
elif sys.argv[3] == "False":
    print("We are not using GPU...")
    import tensorflow as tf
else:
    print("Write literally 'True' or 'False' if you want to use GPU or not.")
    sys.exit()

if float(sys.argv[4]) <= 0 or float(sys.argv[4]) > 1:
    print("Please, indicate a value for the threshold bigger than 0 and 1 as a maximum.")
    sys.exit()

def main():
    input_w, input_h = 416, 416

    interpreter = tf.lite.Interpreter(model_path = sys.argv[1])
    interpreter.resize_tensor_input(interpreter.get_input_details()[0]["index"], [1,input_w,input_h,3], strict = True)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    file = open(sys.argv[2])
    json_dictionary = json.load(file)
    file.close()

    labels = json_dictionary["labels"]
    anchors = json_dictionary["anchors"]

    class_threshold = float(sys.argv[4])

    vid = cv2.VideoCapture(0) 
    
    while(True):
        ret, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image, image_w, image_h = load_image_stream(frame, (input_w, input_h))
        interpreter.set_tensor(interpreter.get_input_details()[0]["index"], image)
        interpreter.invoke()
        predicciones = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        array1, array2, array3 = [predicciones[i] for i in range(len(predicciones))]

        predicciones[0], predicciones[1], predicciones[2] = array2, array3, array1

        boxes = list()
        for i in range(len(predicciones)):
            boxes += decode_netout(predicciones[i][0], anchors[i], class_threshold, input_h, input_w)

        correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
        do_nms(boxes, 0.5)
        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

        image = frame
        for i in range(len(v_boxes)):
            y1, x1, y2, x2 = v_boxes[i].ymin, v_boxes[i].xmin, v_boxes[i].ymax, v_boxes[i].xmax
            if v_labels[i] == "with_mask":
                frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                frame = cv2.putText(frame, v_labels[i]+"    "+str(np.round(v_scores[i], 2)), (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif v_labels[i] == "without_mask":
                frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
                frame = cv2.putText(frame, v_labels[i]+"    "+str(np.round(v_scores[i], 2)), (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (255,165,0), 2)
                frame = cv2.putText(frame, v_labels[i]+"    "+str(np.round(v_scores[i], 2)), (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame', image)
        # Press 'q' to close the windows.
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
     main()