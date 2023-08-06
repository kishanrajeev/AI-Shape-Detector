FROM tensorflow/tensorflow:latest

RUN pip install keras==2.7.0
RUN pip install tensorflow
RUN pip install tensorflow-datasets
