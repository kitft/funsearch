# the python version in the sandbox should be the same as in the host so that cloudpickle works reliably
ARG PYTHON_VERSION=3.11.6

FROM docker.io/python:${PYTHON_VERSION}

ARG INSTALL_PACKAGES
RUN pip install cloudpickle ${INSTALL_PACKAGES}
