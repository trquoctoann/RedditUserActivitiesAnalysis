FROM jupyter/pyspark-notebook

USER root
RUN echo "root:123" | chpasswd
RUN echo "root   ALL=(ALL)       ALL" >> /etc/sudoers

ENV SPARK_HOME /usr/local/spark

RUN wget https://dev.mysql.com/get/Downloads/Connector-J/mysql-connector-j-8.0.32.tar.gz
RUN tar -xzf mysql-connector-j-8.0.32.tar.gz
RUN cp mysql-connector-j-8.0.32/mysql-connector-j-8.0.32.jar $SPARK_HOME
RUN rm mysql-connector-j-8.0.32.tar.gz 

ADD config/spark-defaults.conf $SPARK_HOME/conf/spark-defaults.conf

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 4040 6066 7077 8080 8081 8888