FROM python:3.6-slim-buster

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
COPY requirements.txt /app

# Install dependancies
RUN pip install -r requirements.txt

# copy files over
COPY . /app

# Expose port
EXPOSE 8000

# Run the application
CMD ["gunicorn","wsgi:app","-b 0.0.0.0:8000"]