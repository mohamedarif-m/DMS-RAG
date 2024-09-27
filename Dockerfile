# Use the official Selenium Standalone Chrome image as the base image
FROM registry.access.redhat.com/ubi8/python-311:latest

USER 0
RUN yum install -y java-1.8.0-openjdk

USER 1001
# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the container and install dependencies
COPY requirements.txt /app/requirements.txt
USER 0
RUN pip3 install -r requirements.txt

# Copy your FastAPI Python script to the container
COPY . .

RUN python3 prereqs.py

RUN wget https://repo1.maven.org/maven2/com/ibm/db2/jcc/db2jcc/db2jcc4/db2jcc-db2jcc4.jar
RUN mv db2jcc-db2jcc4.jar db2jcc4.jar

EXPOSE 4050

# Set the command to run your Python script
CMD ["python3", "app.py"]
