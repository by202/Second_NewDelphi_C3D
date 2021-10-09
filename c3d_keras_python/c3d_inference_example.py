from sport1m_model import create_model_functional
import numpy as np
import cv2


def main():

    with open ('keras_labels.txt','r') as f:
        class_names = f.readlines()
        f.close()
        
# =============================================================================
#     Initi model    
# sports1M_weights.h5/ C3D_Sport1M_weights_keras_2.2.4.h5/c3d-pretrained.pth
# =============================================================================

    model = create_model_functional()
    try:
        model.load_weights('c3d-pretrained.pth')
    except OSError as err:
        print('Check path to the model weights\' file!\n\n', err)
        
    model.summary()    
    # lr = 0.005
    # sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    #model.load_weights('./results/weights_c3d.h5', by_name=True)
    
# =============================================================================
#     Read video
# =============================================================================
    video = '12.avi'
    cap = cv2.VideoCapture(video)

    clip = []
    while True:
        ret, frame = cap.read()
        if ret:
            ## convert image to RGB color for matplotlib
            tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip.append(cv2.resize(tmp, (171, 128)))
            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs[..., 0] -= 99.9
                inputs[..., 1] -= 92.1
                inputs[..., 2] -= 82.6
                inputs[..., 0] /= 65.8
                inputs[..., 1] /= 62.3
                inputs[..., 2] /= 60.3
                # center crop  (, l, h, w, c)
                inputs = inputs[:,:,8:120,30:142,:]
                inputs = np.transpose(inputs, (0, 2, 3, 1, 4))

                pred = model.predict(inputs)
                label = np.argmax(pred[0])
                print("label with prob {} is {}".format(class_names[label], pred[0][label]))
                #print('lable',label)
                # with open("result.txt", "a") as outfile:
                #   outfile.write(str(class_names[label])+str(pred[0][label])+'\n') 
                

# =============================================================================
#                 cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
#                              cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                              (0, 0, 255), 1)
#                 cv2.putText(frame, "prob: %.4f" % pred[0][label], (20, 40),
#                              cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                              (0, 0, 255), 1)
# =============================================================================

                clip.pop(0)
            #cv2.imshow('result', frame)
# =============================================================================
#             cv2.waitKey(10)
#         else:
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#      
# =============================================================================
        
        

# =============================================================================
#     # 16 black frames with 3 channels
#     #dummy_input = np.zeros((1, 16, 112, 112, 3))
# 
#     #prediction_softmax = model.predict(dummy_input)
#     #predicted_class = np.argmax(prediction_softmax)
# 
#     #print('{}Success, predicted class index is: {}{}'.format('\033[92m',
#                                                              predicted_class,
#                                                              '\033[0m'))
# =============================================================================


if __name__ == "__main__":
    main()
