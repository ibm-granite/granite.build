
from fastapi.responses import JSONResponse
from fastapi import status, Request
    

def get_header_auth_token(auth_header: str):
    """Get GIT token from request header authorization"""

    if auth_header == "" or not auth_header.startswith("Bearer "):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "detail": "Authorization header is missing/invalid!",
            },
        )
    return auth_header.removeprefix("Bearer ")

    