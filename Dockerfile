#FROM ubuntu:18.04
#FROM jrottenberg/ffmpeg:4.2-vaapi as ffmpeg
FROM ubuntu:18.04
LABEL maintainer "blakeb@blakeshome.com"

#COPY --from=ffmpeg /usr/local /usr/local

ENV DEBIAN_FRONTEND=noninteractive
# Install packages for apt repo
RUN export DEBIAN_FRONTEND=noninteractive; \
    export DEBCONF_NONINTERACTIVE_SEEN=true; \
    apt-get -qq update && apt-get -qqy install --option Dpkg::Options::="--force-confnew" --no-install-recommends \
    tzdata \
    software-properties-common \
    build-essential \
    gnupg wget unzip tzdata \
    # libcap-dev \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    #&& add-apt-repository ppa:savoury1/ffmpeg4 -y \
    #&& add-apt-repository ppa:savoury1/graphics -y \
    #&& add-apt-repository ppa:savoury1/multimedia -y \
    #&& add-apt-repository ppa:ubuntu-toolchain-r/test -y \
    && add-apt-repository ppa:jonathonf/ffmpeg-4 -y \
    && apt -qq install --no-install-recommends -y \
        python3.7 \
        python3.7-dev \
        python3-pip \
        ffmpeg \
        # VAAPI drivers for Intel hardware accel
        libva-drm2 libva2 i965-va-driver vainfo \
    && python3.7 -m pip install -U pip \
    && python3.7 -m pip install -U wheel setuptools \
    && python3.7 -m pip install --upgrade pip \
    && python3.7 -m pip install -U \
        opencv-python-headless \
        # python-prctl \
        numpy \
        imutils \
        scipy \
        psutil \
    && python3.7 -m pip install -U \
        Flask \
        paho-mqtt \
        PyYAML \
        matplotlib \
        pyarrow \
        requests \
        pathlib \
        click \
    && echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" > /etc/apt/sources.list.d/coral-edgetpu.list \
    && wget -q -O - https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt -qq update \
    && echo "libedgetpu1-max libedgetpu/accepted-eula boolean true" | debconf-set-selections \
    && apt -qq install --no-install-recommends -y \
        libedgetpu1-max \
    ## Tensorflow lite (python 3.7 only)
    && wget -q https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl \
    && python3.7 -m pip install tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl \
    && rm tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl \
    && rm -rf /var/lib/apt/lists/* \
    && (apt-get autoremove -y; apt-get autoclean -y)

# get model and labels
RUN wget -q https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite -O /edgetpu_model.tflite --trust-server-names
COPY labelmap.txt /labelmap.txt
RUN wget -q https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess.tflite -O /cpu_model.tflite 


RUN mkdir /cache /clips

WORKDIR /opt/frigate/
ADD frigate frigate/
COPY detect_objects.py .
COPY benchmark.py .
COPY process_clip.py .

ENTRYPOINT ["python3.7", "-u", "detect_objects.py"]
