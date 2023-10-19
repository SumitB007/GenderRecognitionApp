import numpy as np
import sklearn
import pickle
import cv2


# Load all models
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')  # cascade classifier
model_svm = pickle.load(open('./model/model_svm.pickle',mode='rb'))  # machine learning model svm
pca_models = pickle.load(open('./model/pca_dict.pickle',mode='rb')) #pca dictionary


model_pca = pca_models['pca']  # PCA model
mean_face_arr = pca_models['mean_face']  # Mean Face


#Create Pipeline
def faceRecognitionPipeline(filename, path=True):
    if path:
        # step1 read image
        img = cv2.imread(filename) #BGR
    else:
        img = filename # array
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # step2 convert the image into grayscale
    img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)

    # step3 crop the face using haar cascade classifier
    faces = haar.detectMultiScale(img_gray,1.5,3)

    predictions =[]

    for x,y,w,h in faces:
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi = img_gray[y:y+h,x:x+w]
        # plt.imshow(roi,cmap='gray')
        # plt.show()

        # step4 normalization (0-1)
        roi = roi / 255.0

        #step5 resize images(100x100)
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)

        #stpe6 flattening image(1x10000)
        roi_reshape = roi_resize.reshape(1,10000)

        #step7 subtract with mean
        roi_mean = roi_reshape - mean_face_arr #subtracting face with mean face

        #step8 get eigen image (apply pca)
        eigen_image = model_pca.transform(roi_mean)

        #step9 eigen image for visualization
        eig_img = model_pca.inverse_transform(eigen_image)

        #step10 pass to ml model (svm) and get predictions
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()
        print(results,prob_score)

        # step11 generate report
        text = "%s : %d"%(results[0],prob_score_max*100)
        print(text)

        # define colors based on results
        if results[0]=='male':
            color = (255,255,0)
        else:
            color=(255,0,255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color,-1)
        cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),2)

        output ={
            'roi':roi,
            'eig_img':eig_img,
            'prediction_name': results[0],
            'score': prob_score_max
        }

        predictions.append(output)
    return img,predictions




