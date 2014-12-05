import csv
from datetime import *

class TimeSeriesDataFile:
    def __init__(self, file_path, is_standard_format = True):
        self._filePath         = file_path;
        self._isStandardFormat = is_standard_format;

    def ExtractFeatures(self, ignore_columns = set([])):
        with open(self._filePath, 'rb') as f:
            reader = csv.reader(f)
            if (self._isStandardFormat):
                return StandardFileParser(ignore_columns).ParseFile(reader) 

        # Couldn't open file
        return False


class StandardFileParser:
    def __init__(self, ignored_columns, date_format = '%d.%m.%Y %H:%M:%S.%f'):
        self._ignoredColumns = ignored_columns
        self._dateFormat     = date_format

    def ParseFile(self, csv_reader):
        # Get feature names (headers in excel file)
        file_headers = csv_reader.next()

        # Initializes features and mappings from headers
        features = self._InitializeFeatures(file_headers)
        
        # Ignore extraneous lines
        csv_reader.next()
        csv_reader.next()

        for row in csv_reader:
            # parse time
            date = datetime.strptime(row[self._HeaderIndex('Time')], self._dateFormat)

            for feature in features:
                idx = self._HeaderIndex(feature)
                val = float(row[idx])
                features[feature].Append(date, val)

        return features

    def _InitializeFeatures(self, file_headers):
        # Initializes a header to column index mapping
        self._headerToColumnIndexMapping = {}
        for idx, header in enumerate(file_headers):
            self._headerToColumnIndexMapping[header] = idx

        # Returns a list of FeatureTimeSeries
        features = {}
        for header in file_headers:
            if self._ShouldSaveFeature(header):
                features[header] = FeatureTimeSeries(header)
        return features

    def _HeaderIndex(self, header):
        return self._headerToColumnIndexMapping[header]

    def _ShouldSaveFeature(self, feature):
        return feature != 'Time'


class FeatureTimeSeries:
    def __init__(self, name):
        self._name = name
        self._min  = 99999999
        self._max  = 0
        self._data = {}

    def Append(self, date, feature_value):
        if feature_value < self._min:
            self._min = feature_value
        if feature_value > self._max:
            self._max = feature_value
        self._data[date] = feature_value

    def __str__(self):
        return self._data.__str__()
