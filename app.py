import tensorflow.compat.v1 as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
import tempfile
from flask import Flask,render_template,url_for,request
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
	return 'Hello, World :)'

@app.route('/about')
def about():
    return 'Hello, World!<br> This website is out of service.<br>THANK YOU :)'

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

@app.route('/predict', methods=['POST'])
def predict():
   
    return 'Hello, World :)'


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(debug=True,threaded=True, port=5000)