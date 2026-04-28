# This the temporary file name used by all instances to hold a cert file specified in an env var.  
# Since all instances use the same env vars, they can all use the same file.
import atexit
import base64
import os
import threading
from typing import Optional
from gbserver.types.constants import GBSERVER_SQL_SSLROOT_CERT_BASE64, GBSERVER_SQL_SSLROOT_CERT_FILE
from gbserver.utils.logger import LoggingUtility
from gbserver.utils.utils import create_temp_file_name


_SSL_CERT_FILE : Optional[str] = None
_lock = threading.Lock()

def _cleanup():
    with _lock:
        global _SSL_CERT_FILE
        if _SSL_CERT_FILE is not None and os.path.exists(_SSL_CERT_FILE):
            os.remove(_SSL_CERT_FILE) 
            _SSL_CERT_FILE = None 


def get_ssl_cert_file(logger:LoggingUtility) -> Optional[str]:
    """Read the GBSERVERSQL_SSLROOT_CERT_FILE/BASE64 values and return the location of the certificate file.
    If neither are specified, return None. 
    If both are specified, FILE takes precedence.
    If BASE64 is specified but no file is specified, then write decoded text to a temporary file (and arrange for it to be deleted on exit).
    Raise exceptions if file does not exist or is zero length.
    """
    global _SSL_CERT_FILE
    if _SSL_CERT_FILE is not None:
        return _SSL_CERT_FILE
    ssl_cert_file : Optional[str] = GBSERVER_SQL_SSLROOT_CERT_FILE
    ssl_cert_base64= GBSERVER_SQL_SSLROOT_CERT_BASE64
    if ssl_cert_file is not None:
        if ssl_cert_base64 is not None:
            if os.path.isfile(ssl_cert_file):
                logger.warning(f"Both SQL ssl file and env var specified.  Using file specified {ssl_cert_file}")
            else:
                logger.warning(f"SQL ssl file {ssl_cert_file} does not exist.  Will use base64 env var")
                ssl_cert_file = None
    if ssl_cert_file is None and ssl_cert_base64:
        ssl_cert_file = create_temp_file_name(suffix=".cert")
        logger.info(f"Using SQL SSL certification from env var placed in file {ssl_cert_file}")        
        with _lock:
            if os.path.getsize(ssl_cert_file) == 0:
                decoded_data = base64.b64decode(ssl_cert_base64)
                atexit.register(_cleanup)
                with open(ssl_cert_file, 'wb') as file:
                    file.write(decoded_data)

    if ssl_cert_file is not None:
        if not os.path.isfile(ssl_cert_file):
            raise ValueError(f"SQL SSL certificate {ssl_cert_file} does not exist or is not a file")
        elif os.path.getsize(ssl_cert_file) == 0:
            raise ValueError(f"SQL SSL certificate {ssl_cert_file} is empty")
    logger.info(f"SQL SSL cert file located at '{ssl_cert_file}'") 
    _SSL_CERT_FILE = ssl_cert_file
    return _SSL_CERT_FILE   