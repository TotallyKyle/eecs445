import csv
from datetime import *
from enum     import Enum;


#
#
#
class TimeSeriesDataFile:

    # Possible file formats
    class Formats (Enum):
        Standard  = 1
        Alternate = 2

    def __init__(self, file_path, format):
        self._filePath = file_path;
        self._format   = format;

    #
    #
    #
    def ExtractAllFeatures(self):
        with open(self._filePath, 'rb') as f:
            reader = csv.reader(f)
            if (self._format == TimeSeriesDataFile.Formats.Standard):
                return StandardFileParser().ParseFile(reader) 
            else:
                return AlternateFileParser().ParseFile(reader)

#
#
#
class FeatureTimeSeries:
    def __init__(self, name):
        self._data = {}

        # Publicly accessible
        self.FeatureName = name
        self.Min  = 99999999
        self.Max  = 0

    def Append(self, date, feature_value):
        if feature_value < self.Min:
            self.Min = feature_value
        if feature_value > self.Max:
            self.Max = feature_value
        self._data[date] = feature_value

    def __str__(self):
        return self._data.__str__()


########## IMPLEMENTATION ##########

class StandardFileParser:
    def __init__(self, ignored_columns = set([]), date_format = '%d.%m.%Y %H:%M:%S.%f'):
        self._ignoredColumns = ignored_columns
        self._dateFormat     = date_format

    def ParseFile(self, csv_reader):
        # Get feature names (headers in excel file)
        file_headers = csv_reader.next()

        # Initializes features and mappings from headers
        features = self._InitializeFeatures(file_headers)

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




class AlternateFileParser:
    def __init__(self, ignored_columns = set([]), date_format = '%m/%d/%y'):
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
            for feature in features:
                feature_idx = self._HeaderColumn(feature)
                time_str    = row[feature_idx]
                value_str   = row[feature_idx + 1]
                if time_str != '' and value_str != '':
                    time = datetime.strptime(time_str, self._dateFormat)
                    value = float(value_str)
                    features[feature].Append(time, value)
        return features

    def _InitializeFeatures(self, file_headers):
        # Initializes a mapping from header to index
        self._headerToColumnIndexMapping = {}
        for idx, feature in enumerate(file_headers):
            if feature != "":
                self._headerToColumnIndexMapping[feature] = idx

        # Returns a map of FeatureTimeSeries
        features = {}
        for feature in file_headers:
            if feature != "":
                features[feature] = FeatureTimeSeries(feature)
        return features

    def _HeaderColumn(self, header):
        return self._headerToColumnIndexMapping[header]
