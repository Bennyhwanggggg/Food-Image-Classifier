FROM python:3.7

RUN apt-get update && apt-get install -y \
  nginx \
  build-essential \
	python \
	python-pip \
	&& rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

WORKDIR src/
COPY ./* ./

RUN pip install -r requirements_web.txt

EXPOSE 5000

ENV AWSKEY=$AWSKEY
ENV AWSSECRET=$AWSSECRET

COPY nginx.conf /etc/nginx/conf.d/default.conf
RUN chmod +x ./start.sh
CMD ["./start.sh"]