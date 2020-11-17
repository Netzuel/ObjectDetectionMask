import numpy as np
import tensorflow as tf
import sys
import shutil

if len(sys.argv) != 4:
    print("Please, insert 3 arguments:\n1) Input model type, that is, 'pb' or 'h5' depending on what you want to transform.\n2) Path to the model. Remember to also specify the format, that is, '.pb' or '.h5' depending on what you specified in the first argument.\n3) Path for the output. Remember to specify '.tflite' since it has to be a Tensorflow Lite model.")
    sys.exit()

model_type = sys.argv[1]
input_path = sys.argv[2]
output_path = sys.argv[3]

print("NumPy version: ", np.__version__)
print("Tensorflow version: ", tf.__version__)

if model_type == "pb":
    model = tf.saved_model.load(input_path)
elif model_type == "h5":
    model = tf.keras.models.load_model(input_path)
else:
    print("Please, specify, literally, 'pb' or 'h5' depending on what format you have your input model in.")
    sys.exit()

export_dir = "saved_model"
tf.saved_model.save(model, export_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

tflite_model_file = output_path

try:
    with open(tflite_model_file, "wb") as f:
        f.write(tflite_model)
        shutil.rmtree("saved_model", ignore_errors=True)
        print("Compiled succesfully. Your .tflite file should be in the output path you specified.")
except:
    print("An error happened: Pay attention to the output path. Make sure you write 'your_model_name.tflite'.")