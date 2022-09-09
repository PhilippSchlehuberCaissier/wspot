FROM debian:sid

# We have a lot of timeout issues when fetching packages from
# http://httpredir.debian.org/debian
RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get update
RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get upgrade -y
RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y wget gnupg apt-utils

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get update

RUN apt-get update && \
    RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
      python3-pip python3-dev python3-setuptools python3-matplotlib python3-numpy python3-pandas \
      build-essential git graphviz fonts-lato \
      sudo unzip imagemagick ghostscript libbdd0c2 \
      man-db less bison \
      openjdk-11-jre-headless && \
    apt-get clean

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends npm
RUN rm -rf /var/lib/apt/lists/* && \
    set -x && \
    pip3 install --no-cache-dir jupyter ipywidgets jupyterlab && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
    rm -rf ~/.cache

RUN echo "deb http://deb.debian.org/debian/ sid main" >> /etc/apt/sources.list
RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get upgrade -y

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y flex cmake bison \
    libboost-all-dev doxygen 

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

RUN cd ${HOME} && git clone https://github.com/catchorg/Catch2.git && cd Catch2 && git checkout v2.13.5\
    && mkdir build && cd build && cmake .. && make -j 8 && make install

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y swig

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y texlive-base

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y latexmk

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y texlive-extra-utils

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y texlive-latex-extra

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y texlive-fonts-extra

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y texlive-science


RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y pdf2svg

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y clang++-13

RUN mkdir ${HOME}/code
COPY build_all.sh ${HOME}/code/build_all.sh
RUN cd ${HOME}/code && ./build_all.sh "-DCMAKE_CXX_COMPILER=clang++-13" "CXX=clang++-13"
##
COPY install1.sh /tmp/install1.sh
COPY custom.css ${HOME}/.jupyter/custom/custom.css
COPY custom.js ${HOME}/.jupyter/custom/custom.js
##
# 2017-01-19: Today's version of Jupyter takes custom.js from
# ~/.jupyter, and custom.css from ~/.ipython.

RUN ln -s /usr/share/fonts/truetype/lato/Lato-Regular.ttf ${HOME}/.jupyter/custom/Lato-Regular.ttf \
  && mkdir -p ${HOME}/.ipython \
  && ln -s ${HOME}/.jupyter/custom ${HOME}/.ipython/custom \
  && cd /tmp && ./install1.sh && rm -f install1.sh

RUN python3 -m pip install --no-cache-dir notebook jupyterlab
RUN pip install --no-cache-dir jupyterhub

#todo put this in build_all
USER root
RUN cp -r --symbolic-link /usr/local/lib/python3.10/site-packages/* /usr/local/lib/python3.10/dist-packages/
RUN cp -r --symbolic-link /usr/local/lib/* /usr/lib/

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y python3-matplotlib

USER root
ADD tchecker_examples ${HOME}/code/tchecker_examples
COPY simple_1CTA.py ${HOME}/code/simple_1CTA.py
COPY to_weighted_twa.py ${HOME}/code/to_weighted_twa.py
COPY WBA_utils.py ${HOME}/code/WBA_utils.py
COPY one_exec.py ${HOME}/code/one_exec.py
COPY run_bench.py ${HOME}/code/run_bench.py
COPY energy_buechi.ipynb ${HOME}/code/energy_buechi.ipynb
COPY energy_buechi_bench.ipynb ${HOME}/code/energy_buechi_bench.ipynb

RUN chown -R ${NB_UID} ${HOME}

RUN python3 -m pip install parse

WORKDIR ${HOME}/code
USER ${NB_USER}

RUN echo "Extensions"
RUN jupyter nbextension enable --py widgetsnbextension
RUN echo "DONE!"
