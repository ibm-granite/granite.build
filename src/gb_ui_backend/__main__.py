import uvicorn


def main():
    uvicorn.run(
        "gb_ui_backend.main:app",
        host="0.0.0.0",
        port=8090,
        reload=False,
    )


if __name__ == "__main__":
    main()
