FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent.py .
RUN useradd --uid 10001 --create-home agent && mkdir /app/output && chown agent:agent /app/output
USER agent
ENTRYPOINT ["python", "agent.py"]
CMD ["--demo"]
