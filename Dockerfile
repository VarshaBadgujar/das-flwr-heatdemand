FROM nvcr.io/nvidia/tensorflow:23.12-tf2-py3
RUN pip install "flwr[simulation]==1.27.0" xgboost scikit-learn \
    pyarrow pyyaml scipy --quiet
WORKDIR /workspace
