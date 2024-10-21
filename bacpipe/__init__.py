import logging

# Initialize Logger
logger = logging.getLogger('bacpipe')
c_handler = logging.StreamHandler()
c_format = logging.Formatter(
    '%(name)s::%(levelname)s:%(message)s'
    )
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
logger.setLevel(logging.DEBUG)