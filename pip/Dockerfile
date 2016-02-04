FROM ruimashita/python

MAINTAINER takuya.wakisaka@moldweorp.com

# install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py

# Fix: InsecurePlatformWarning
# http://urllib3.readthedocs.org/en/latest/security.html#insecureplatformwarning
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/ \
    && pip install --no-cache-dir ndg-httpsclient


