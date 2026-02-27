FROM pytorch/pytorch:2.1.2-cpu-runtime

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit","run","app.py","--server.port=8051","--server.address=0.0.0"]