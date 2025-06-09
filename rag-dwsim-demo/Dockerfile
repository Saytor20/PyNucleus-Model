# Base on a Linux+Mono image (includes Mono runtime)
FROM mono:6.12.0

# Install Python3 & pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip

# Install pythonnet + Excel libraries
RUN pip3 install --upgrade pip \
    && pip3 install pythonnet pandas openpyxl

# Copy the DWSIM DLLs into Mono's GAC folder in the container
COPY libs/DWSIM.Thermodynamics.dll /usr/local/lib/mono/4.5/
COPY libs/DWSIM.Interfaces.dll     /usr/local/lib/mono/4.5/

# Copy your test script in
WORKDIR /app
COPY dwsim_test.py .

# When the container runs, execute your test
ENTRYPOINT ["python3", "dwsim_test.py"]
