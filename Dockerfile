# syntax=docker/dockerfile:1
FROM python:3.8 
#ansonxing168/sentencepiece
#FROM huahua123ldh/pytorch:gpu-1.2-torchtext-sp-tsfm
#FROM huggingface/transformers-tensorflow-gpu
COPY requirements.txt .
#VOLUME /model
#RUN wget https://github.com/google/sentencepiece.git
#RUN apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
#RUN cd sentencepiece
#RUN mkdir build && cd build
#RUN apt-get install cmake
#RUN cmake ..
#RUN make -j $(nproc)
#RUN sudo make install
#RUN sudo ldconfig -v
RUN pip install -r requirements.txt --no-cache-dir
COPY src/ . 
CMD [ "python", "./slt_server.py" ]
