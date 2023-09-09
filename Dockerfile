FROM python:3.10

RUN apt-get update
RUN apt-get install -y libsndfile1 \
  python3 \
  python3-pip \
  locales

RUN localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm-256color

WORKDIR /app

COPY ./requirements.txt ./requirements.txt
COPY entrypoint.sh /usr/bin/
RUN chmod +x /usr/bin/entrypoint.sh
RUN pip install -r requirements.txt

ENTRYPOINT ["entrypoint.sh"]
EXPOSE 8000