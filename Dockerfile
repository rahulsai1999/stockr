FROM python:3.7.3-slim
WORKDIR /app
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --upgrade tensorflow==1.14 --no-cache-dir
RUN pip install gevent
COPY . .
EXPOSE 5000
CMD ["gunicorn","-t","120","-b","0.0.0.0:5000","app:app","-k","gevent"]
