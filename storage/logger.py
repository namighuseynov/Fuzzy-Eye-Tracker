# io/logger.py
import csv

class CSVLogger:
    def __init__(self, path):
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.wr = csv.writer(self.f)

    def write_header(self, header):
        self.wr.writerow(header)

    def write_row(self, row):
        self.wr.writerow(row)

    def close(self):
        self.f.close()
