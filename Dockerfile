FROM nvidia/cuda:10.2-cudnn7-devel
RUN apt update
RUN apt install python3 python3-pip ffmpeg -y
RUN pip3 install torch torchvision
RUN pip3 install numpy tqdm opencv-python
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
