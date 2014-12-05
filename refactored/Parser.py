import csv


class FeaturesFile:

    def __init__(self, file_path, is_standard_format = True):
        self._filePath         = file_path;
        self._isStandardFormat = is_standard_format;

    def ParseFile(self):
        with open(self._filePath, 'rb') as f:
            reader = csv.reader(f)
            if (self._isStandardFormat):
                return self._ParseFileStandardFormat(reader)
            else:
                return self._ParseFileAlternateFormat(reader)

        # Couldn't open file
        return False

    def _ParseFileStandardFormat(self, csv_reader):
        line = csv_reader.next()
        print(line)
        return False

    def _ParseFileAlternateFormat(self, csv_reader):
        return False
