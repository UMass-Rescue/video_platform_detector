from tf_object_detection_util.training_api import train

def trainObjDetectorTest():
    trainDir = 'data/tik_tok/images/new_expanded_dataset/all/'
    predictDir = 'tf_test_out'
    fineTunedModelPath = '/home/prasannals/models/research/object_detection/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt'
    tfObjectDetFolder = '/home/prasannals/models/research/object_detection'
    # o.labelObjects(trainDir, predictDir, fineTunedModelPath, tfObjectDetFolder)
    train(trainDir, fineTunedModelPath, tfObjectDetFolder, destn=predictDir)


trainObjDetectorTest()
# Interrupt the kernel to stop training. This will also automatically create and save the inference graph