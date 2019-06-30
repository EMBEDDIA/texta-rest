#!/bin/bash

su -c ". /var/miniconda3/bin/activate texta-rest && python migrate.py" -m -s /bin/bash www-data

# Get the maximum upload file size for Nginx, default to 0: unlimited
USE_NGINX_MAX_UPLOAD=${NGINX_MAX_UPLOAD:-0}
# Generate Nginx config for maximum upload file size
echo "client_max_body_size $USE_NGINX_MAX_UPLOAD;" > /etc/nginx/conf.d/upload.conf

# Get the number of workers for Nginx, default to 1
USE_NGINX_WORKER_PROCESSES=${NGINX_WORKER_PROCESSES:-1}
# Modify the number of worker processes in Nginx config
sed -i "/worker_processes\s/c\worker_processes ${USE_NGINX_WORKER_PROCESSES};" /etc/nginx/nginx.conf

# Set the max number of connections per worker for Nginx, if requested
# Cannot exceed worker_rlimit_nofile, see NGINX_WORKER_OPEN_FILES below
if [[ -v NGINX_WORKER_CONNECTIONS ]] ; then
    sed -i "/worker_connections\s/c\    worker_connections ${NGINX_WORKER_CONNECTIONS};" /etc/nginx/nginx.conf
fi

# Set the max number of open file descriptors for Nginx workers, if requested
if [[ -v NGINX_WORKER_OPEN_FILES ]] ; then
    echo "worker_rlimit_nofile ${NGINX_WORKER_OPEN_FILES};" >> /etc/nginx/nginx.conf
fi

# HACK: fix cron (sometimes pam_loginuid causes problems)
sed -i -e 's/\(session.*pam_loginuid.so\)/#\1/' /etc/pam.d/cron

set | egrep "^(DJANGO|TEXTA|PYTHON|LC_|LANG)" | sed -e 's/^/export /' > .env

exec "$@"
