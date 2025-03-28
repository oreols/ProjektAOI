import csv
import os

class CSVLogger:
    def __init__(self, path, headers):
        self.path = path
        self.headers = headers
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log(self, values):
        with open(self.path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(values)
