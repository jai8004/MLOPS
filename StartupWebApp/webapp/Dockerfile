 
#python as base image
FROM python:3.6-slim

#working directory /app
WORKDIR /app

#Copy all the content of current directory to /app
ADD . /app

#Installing required packages
RUN pip3 install -r requirements.txt

#Open port 5000
EXPOSE 5000

#Set environment variable
ENV NAME env

#Run python program
CMD ["python","app.py"]
