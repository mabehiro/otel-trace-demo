FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /projects

RUN chgrp -R 0 /projects && chmod -R g=u /projects

COPY requirements.txt ./

# Copy the content of the local src directory to the working directory
COPY . .

# Install any dependencies
RUN \
  if [ -f requirements.txt ]; \
    then pip install --no-cache-dir -r requirements.txt; \
  elif [ `ls -1q *.txt | wc -l` == 1 ]; \
    then pip install -r *.txt; \
  fi

# Specify the command to run on container start
CMD [ "python3", "./agentic-chatbot.py" ]
