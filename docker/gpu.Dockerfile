FROM nvidia/cuda:11.1.1-base-ubuntu20.04

# Install system packages
RUN set -x \
    && apt-get update && apt-get install cmake build-essential wget -y \
    # apt clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && rm Miniconda3-latest-Linux-x86_64.sh

# Add conda binaries to path
ENV PATH /opt/conda/bin:$PATH

# Copy env file
COPY environment-gpu.yaml /var/environment.yaml
# Change env name to texta-rest so we can use conf files from CPU version
RUN sed -i "1s/.*/name: texta-rest/" /var/environment.yaml
# Create env
RUN conda env create -f /var/environment.yaml
ENV PATH /var/miniconda3/envs/texta-rest/bin:$PATH

# clean conda
RUN conda clean --all -y

# Copy project files
COPY . /var/texta-rest

# Retrieve pre-built front
RUN wget https://packages.texta.ee/texta-rest-front/texta-rest-front-latest.tar.gz \
    && tar -zxvf texta-rest-front-latest.tar.gz \
    && cp -r dist/TEXTA /var/texta-rest/front \
    && rm texta-rest-front-latest.tar.gz && rm -R dist

# Ownership to www-data and entrypoint
RUN chown -R www-data:www-data /var/texta-rest \
    && chmod 775 -R /var/texta-rest \
    && chmod 777 -R /opt/conda/envs/texta-rest/var \
    && chmod +x /var/texta-rest/docker/conf/entrypoint.sh \
    && rm -rf /var/texta-rest/.git \
    && rm -rf /root/.cache

# System configuration files
COPY docker/conf/supervisord.conf /opt/conda/envs/texta-rest/etc/supervisord/conf.d/supervisord.conf
COPY docker/conf/nginx.conf /opt/conda/envs/texta-rest/etc/nginx/sites.d/default-site.conf
ENV UWSGI_INI /var/texta-rest/docker/conf/texta-rest.ini

# Set environment variables
ENV JOBLIB_MULTIPROCESSING 0
ENV PYTHONIOENCODING=UTF-8
ENV UWSGI_CHEAPER 2
ENV UWSGI_PROCESSES 16

# Expose ports
EXPOSE 80
EXPOSE 8000
EXPOSE 8001
EXPOSE 8080

# change dir & run collectstatic
WORKDIR /var/texta-rest

# Do not add resources do the image.
ENV SKIP_MLP_RESOURCES True
ENV SKIP_BERT_RESOURCES True
RUN /opt/conda/envs/texta-rest/bin/python manage.py collectstatic --no-input --clear
# set back to False
ENV SKIP_MLP_RESOURCES False
ENV SKIP_BERT_RESOURCES False

# Ignition!
ENTRYPOINT ["/var/texta-rest/docker/conf/entrypoint.sh"]
CMD ["supervisord", "-n"]
