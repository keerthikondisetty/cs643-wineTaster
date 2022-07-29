FROM java:8-jdk

RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o get-pip.py \
    && python get-pip.py \
    && pip install pyspark \
    && mkdir ~/cs643

COPY predicting.py ~/cs643/
COPY training.py ~/cs643/
COPY TrainingDataset.csv ~/cs643/

WORKDIR ~/CS643/

CMD python training.py
RUN cd ~/cs643
ENTRYPOINT ["python","predicting.py"]

