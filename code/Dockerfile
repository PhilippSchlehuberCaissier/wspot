FROM debian:sid

# We have a lot of timeout issues when fetching packages from
# http://httpredir.debian.org/debian
#RUN echo "deb http://deb.debian.org/debian/ sid main" >> /etc/apt/sources.list

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get update

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y wget gnupg apt-utils

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get upgrade -y

RUN apt-get update && \
    RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
      python3-pip python3-dev python3-setuptools python3-matplotlib python3-numpy python3-pandas \
      build-essential git graphviz fonts-lato \
      sudo unzip imagemagick ghostscript libbdd0c2 \
      man-db less bison \
      openjdk-11-jre-headless flex cmake bison \
      libboost-all-dev doxygen swig\
      texlive-base latexmk texlive-extra-utils texlive-latex-extra \
      texlive-fonts-extra texlive-science pdf2svg clang++-13 python3-matplotlib&& \
    apt-get clean

RUN RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends npm
RUN rm -rf /var/lib/apt/lists/* && \
    set -x && \
    pip3 install --no-cache-dir jupyter ipywidgets jupyterlab && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
    rm -rf ~/.cache


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


RUN mkdir ${HOME}/code
COPY build_all.sh ${HOME}/code/build_all.sh
RUN cd ${HOME}/code && ./build_all.sh "-DCMAKE_CXX_COMPILER=clang++-13" "CXX=clang++-13"
##
COPY install1.sh /tmp/install1.sh
COPY custom/* ${HOME}/.jupyter/custom/
##
# 2017-01-19: Today's version of Jupyter takes custom.js from
# ~/.jupyter, and custom.css from ~/.ipython.

RUN ln -s /usr/share/fonts/truetype/lato/Lato-Regular.ttf ${HOME}/.jupyter/custom/Lato-Regular.ttf \
  && mkdir -p ${HOME}/.ipython \
  && ln -s ${HOME}/.jupyter/custom ${HOME}/.ipython/custom \
  && cd /tmp && ./install1.sh && rm -f install1.sh

RUN python3 -m pip install --no-cache-dir notebook jupyterlab jupyterhub parse

#todo put this in build_all
USER root
RUN cp -r --symbolic-link /usr/local/lib/python3.10/site-packages/* /usr/local/lib/python3.10/dist-packages/
RUN cp -r --symbolic-link /usr/local/lib/* /usr/lib/

USER root
ADD tchecker_examples ${HOME}/code/tchecker_examples
COPY *.py ${HOME}/code/
COPY *.ipynb ${HOME}/code/

RUN chown -R ${NB_UID} ${HOME}

WORKDIR ${HOME}/code
USER ${NB_USER}

RUN echo "Extensions"
RUN jupyter nbextension enable --py widgetsnbextension
RUN echo "DONE!"
