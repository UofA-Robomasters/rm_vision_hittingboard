# rm_vision_hittingboard

## Number Identification nearly done

### cnn_training.py
A Covolutional Network is trained with 98% accuary on test set. Two covolution layers and two fully connected layers are used. The parameters, model, graph would take about 40 MB space.

### number_identify.py
The images must be strengthened before being passed to model. The `process_image` function is incompete, and it will be finished after the camera is bought. Most operations will depends on the camera to make photo more like training data.
The speed of identification has not been test as I do not have enough test images. By experience, it should be pretty quick.
