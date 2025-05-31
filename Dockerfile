FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl wget ca-certificates unzip \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    openjdk-11-jre \
    sysstat \
    && rm -rf /var/lib/apt/lists/*

# Set default Python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1


# Create and activate virtual environment
ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Upgrade pip and install Python packages
RUN pip install --upgrade pip
RUN pip install \
    torch==2.7 \
    fastapi \
    uvicorn[standard] \
    pydantic \
    peft \
    sentencepiece \
    "protobuf<4.0.0" \
    transformers \
    huggingface_hub \
    codecarbon \
    arize-phoenix \
    opentelemetry-api \
    opentelemetry-sdk \
    arize-phoenix-otel \
    opentelemetry-exporter-otlp \
    opentelemetry-instrumentation-fastapi

# Create working directory
WORKDIR /app

# Copy app code
COPY ./app /app

# Install Apache JMeter
ENV JMETER_VERSION=5.6.3
RUN wget https://dlcdn.apache.org//jmeter/binaries/apache-jmeter-${JMETER_VERSION}.tgz && \
    tar -xzf apache-jmeter-${JMETER_VERSION}.tgz && \
    mv apache-jmeter-${JMETER_VERSION} /opt/jmeter && \
    rm apache-jmeter-${JMETER_VERSION}.tgz
ENV PATH="/opt/jmeter/bin:$PATH"

# Install JMeter Plugins Manager and Ultimate Thread Group
RUN curl -L -o /opt/jmeter/lib/ext/JMeterPlugins-Manager.jar https://jmeter-plugins.org/get/ && \
    curl -L -o /opt/jmeter/lib/cmdrunner-2.3.jar http://search.maven.org/remotecontent?filepath=kg/apc/cmdrunner/2.3/cmdrunner-2.3.jar && \
    java -cp "/opt/jmeter/lib/ext/JMeterPlugins-Manager.jar:/opt/jmeter/lib/*" \
    org.jmeterplugins.repository.PluginManagerCMDInstaller && \
    /opt/jmeter/bin/PluginsManagerCMD.sh install tilln-sshmon,jpgc-casutg,jpgc-json,jpgc-perfmon


# Create emissions log directory to prevent CodeCarbon errors
RUN mkdir -p /app/emissions_logs

# Copy and set entrypoint
RUN chmod +x /app/entrypoint.sh

# Expose FastAPI port
EXPOSE 8000

# Entrypoint
ENTRYPOINT ["/bin/bash"]