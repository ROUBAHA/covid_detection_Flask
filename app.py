import tensorflow.compat.v1 as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import tempfile
from flask import Flask,render_template,url_for,request
from PIL import Image
from tensorflow import keras

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',result="none",accurcy=0)
#API : Deep learning system to screen coronavirus disease 2019 pneumonia <br>created by : Khadija Ouchatti <br> Azzedine Hilali <br>  Abdellah Sabbari <br> 
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    dirLab = "static/retrained_labels.txt"
    dirGgraph = "static/retrained_graph.pb"
    predict = 0
    if request.method == 'POST':
 
        image = request.files['namequery']

        pil_image = Image.open(image)

        imgTf = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
    classifications = []

    for stp in tf.gfile.GFile(dirLab):

        classification = stp.rstrip()

        classifications.append(classification)

    with tf.gfile.FastGFile(dirGgraph, 'rb') as retrainedGraphFile:

        graphStr = tf.GraphDef()

        graphStr.ParseFromString(retrainedGraphFile.read())
       
        _ = tf.import_graph_def(graphStr, name='')
    # end with

    with tf.Session() as sess:

        finens = sess.graph.get_tensor_by_name('final_result:0')

        predictt = sess.run(finens, {'DecodeJpeg:0': imgTf})

    
        output = predictt[0].argsort()[-len(predictt[0]):][::-1]


        boolean = True
 
        for prdct in output:
            strc = classifications[prdct]

            if strc.endswith("s"):
                strc = strc[:-1]

            confidence = predictt[0][prdct]

            if boolean:

                accy = confidence * 100.0
                boolean = False
                predict = strc
    return render_template('index.html', result = predict, accurcy = "{0:.2f}".format(accy))

@app.route('/predict2' , methods=['POST'])
def predict2():

    dirLab = "static/retrained_labels.txt"
    dirGgraph = "static/retrained_graph.pb"
    predict = 0
    if request.method == 'POST':
 
        image = request.files['namequery']

        #pil_image = Image.open(image)
        imgTf=img_to_array(load_img(image))
        #imgTf = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
    classifications = []

    for stp in tf.gfile.GFile(dirLab):

        classification = stp.rstrip()

        classifications.append(classification)

    with tf.gfile.FastGFile(dirGgraph, 'rb') as retrainedGraphFile:

        graphStr = tf.GraphDef()

        graphStr.ParseFromString(retrainedGraphFile.read())
       
        _ = tf.import_graph_def(graphStr, name='')
    # end with

    with tf.Session() as sess:

        finens = sess.graph.get_tensor_by_name('final_result:0')

        predictt = sess.run(finens, {'DecodeJpeg:0': imgTf})

    
        output = predictt[0].argsort()[-len(predictt[0]):][::-1]


        boolean = True
 
        for prdct in output:
            strc = classifications[prdct]

            if strc.endswith("s"):
                strc = strc[:-1]

            confidence = predictt[0][prdct]

            if boolean:

                accy = confidence * 100.0
                boolean = False
                predict = strc

    return "{result:"+predict+",accurcy:"+"{0:.2f}".format(accy)+"}"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(debug=True,threaded=True, port=5000)