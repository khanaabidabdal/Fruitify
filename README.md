
# Fruitify : Know Your Fruits

The following repositry consist of the files of the project "Fruitify : Know Your Fruits". It an application designed to classify fruits using Deep Learning models. Fruitify can classify the 100 different types of the fruits with 80% Accuracy. It is also deployed at the Hugging Face, you can check it out using this link : [Fruitify.](https://huggingface.co/spaces/khanaabidabdal/fruitify)

The Dataset used for training the Deep Learning Model was taken from the [Kaggle.](https://www.kaggle.com/datasets/marquis03/fruits-100) It consists of 100 different types of fruit images, each type having 400 images in training dataset. The dataste also include separate files for testing and validation. You can check which fruits it can classify by viewing *'class_names.txt'*.

Here, I used a transfer learning for the development of the this application. The model I used is ResNet50. I changed ResNet50's fully connected (fc) layer and added some Dropout and ReLU functions to address overfitting. Before finalizing ResNet50, I tried some other models too, like VGG, EfficientNetB1, ResNet101, and GoogleNet.


For development of this model, I wrote *'app.py'*, *'model.py'*, *'requirements.txt'*, '*class_names.txt'* and downlaoded the state_dict of the trained model and saved it as the ResNet_Model. 

