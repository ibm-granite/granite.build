class PrintLogger:
    def __init__(self, logger):
        self.logger = logger
        self.buffer = ""

    def write(self, message):
        # Add message to buffer
        self.buffer += message

        # Process complete lines in buffer
        if "\n" in self.buffer:
            lines = self.buffer.split("\n")
            # Keep the last piece if it doesn't end with newline
            self.buffer = lines.pop()

            # Log each complete line
            # stacklevel=2 skips past this write() frame so the logging
            # framework attributes the record to the actual caller instead
            # of print_logger.py.
            for line in lines:
                if line.strip():
                    self.logger.info(line.strip(), stacklevel=2)

    def flush(self):
        # Log any remaining content in buffer when flush is called
        if self.buffer.strip():
            self.logger.info(self.buffer.strip(), stacklevel=2)
            self.buffer = ""

    def isatty(self):
        return False
