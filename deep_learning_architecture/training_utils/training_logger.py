import sys
# =========================
# Logging (unchanged)
# =========================
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def log(message, logger=None):
    if logger:
        logger.write(message + '\n')
    else:
        print(message)
