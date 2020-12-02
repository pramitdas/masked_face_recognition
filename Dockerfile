from ubuntu:18.04
run apt-get update
run apt install vim -y
run apt install python3.7 -y
run apt install python3-pip -y
run pip3 install --upgrade pip
run pip3 install numpy pandas keras
run pip3 install tensorflow==2.2
run pip3 install opencv-python
run pip3 install pillow
run pip3 install imutils
run pip3 install scikit-learn
run pip3 install matplotlib
run pip3 install --upgrade tensorflow
RUN apt install libgl1-mesa-glx -y
copy . /root/
#cmd bash /root/run.sh

