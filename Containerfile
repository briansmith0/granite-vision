FROM registry.access.redhat.com/ubi9
RUN dnf install -y git python3.12 python3.12-pip

RUN pip3.12 install git+https://github.com/huggingface/transformers
RUN pip3.12 install torch pillow

#Download model
RUN echo -e 'from transformers import AutoProcessor, AutoModelForVision2Seq \n\
model_path = "ibm-granite/granite-vision-3.1-2b-preview" \n\
processor = AutoProcessor.from_pretrained(model_path) \n\
model = AutoModelForVision2Seq.from_pretrained(model_path)' | python3.12

COPY granite_vision.py /

CMD ["python3.12", "granite_vision.py"]
