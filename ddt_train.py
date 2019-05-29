import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import tensorflow as tf
import cv2


def pca_features(features,sess):
    k=features.shape.as_list()[0]*features.shape.as_list()[1]*features.shape.as_list()[2]
    print(k)
    features=tf.transpose(features,[0,3,2,1])

    f=tf.reduce_sum(features,reduction_indices=2)
    f=tf.reduce_sum(f,reduction_indices=2)
    f=tf.reduce_sum(f,reduction_indices=0)
    f=tf.divide(f,k)
    x_mean=tf.expand_dims(tf.expand_dims(tf.expand_dims(f,0),2),2)
    features=tf.subtract(features,x_mean)
    reshaped_temp=tf.reshape(features,(features.shape[0],features.shape[1],-1))
    reshaped_temp=tf.transpose(reshaped_temp,[1,0,2])
    reshaped_features=tf.reshape(reshaped_temp,(reshaped_temp.shape[0],-1))

    cov=tf.matmul(reshaped_features,tf.transpose(reshaped_features))
    cov=tf.divide(cov,k)
    eigval,eigvec=tf.self_adjoint_eig(cov)
    real_value=eigval.eval(session=sess)
    first_compo=eigvec[:,-1]

    projected_map=tf.reshape(tf.matmul(tf.expand_dims(first_compo,0),reshaped_features),(1,features.shape[0],-1))
    projected_map=tf.reshape(projected_map,(features.shape[0],features.shape[2],features.shape[3]))

    maxv=tf.reduce_max(projected_map)
    minv=tf.reduce_min(projected_map)
    temp=tf.divide((maxv+minv),tf.abs(maxv+minv))
    result=tf.multiply(projected_map,temp)
    return result 
def _normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalizes an image."""
    image = tf.to_float(image)
    return tf.div(tf.div(image, 255.) - mean, std)

def get_data(path):
    image_list=[]
    if os.path.isdir(path):
        for _file in os.listdir(path):
            temp_path=os.path.join(path,_file)
            if os.path.isfile(temp_path):
                image_list.append(temp_path)
    return image_list
        
    
def main():
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    image_list=get_data('./data')
    base_model=VGG19(weights='imagenet',include_top=False)
    model=Model(inputs=base_model.input,outputs=base_model.get_layer('block5_conv4').output)
    datas=[]
    for img_path in image_list:
        img=image.load_img(img_path,target_size=(224,224))
        x=_normalize(img)
        x=tf.expand_dims(x,axis=0)
        x=x.eval(session=sess)
        datas.append(x)
    x=tf.concat(datas,0)
    x=x.eval(session=sess)
    block5_conv4_features=model.predict(x)
    features=tf.convert_to_tensor(block5_conv4_features)
    projected_map=pca_features(features,sess)
    projected_map=tf.clip_by_value(projected_map,0,255)
    maxv=tf.reshape(projected_map,(projected_map.shape[0],-1))
    maxv=tf.reduce_max(maxv,axis=1,keepdims=True)[0]
    maxv=tf.expand_dims(maxv,axis=1)
    maxv=tf.expand_dims(maxv,axis=1)
    projected_map=tf.divide(projected_map,maxv)
    projected_map=tf.expand_dims(projected_map,1)
    projected_map=tf.transpose(projected_map,[0,2,3,1])
    projected_map=tf.image.resize_bilinear(projected_map,size=[224,224],align_corners=False)
    projected_map=tf.multiply(projected_map,255.0)
    for i,img_path in enumerate(image_list):
        img=[]
        mask=[]
        img = cv2.resize(cv2.imread(img_path), (224, 224))
        projected_map_temp=tf.transpose(projected_map[i],[2,1,0])
        mask=tf.tile(projected_map_temp,[3,1,1])
        mask=tf.transpose(mask,[1,2,0])
        mask=mask.eval(session=sess)
        #print(mask)
 #       print(mask.shape)
#        print(img.shape)
        mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
        save_img = cv2.addWeighted(img, 0.5, mask, 0.5, 0.0)
    #save_imgs = np.concatenate(save_imgs, 1)
        cv2.imwrite('./result/'+str(i)+'.jpg', save_img)
        
if __name__=='__main__':
    main()

