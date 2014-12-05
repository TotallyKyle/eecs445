from TimeSeriesDataFile import TimeSeriesDataFile

f  = TimeSeriesDataFile('USDJPY_complete.csv', TimeSeriesDataFile.Formats.Standard)
f2 = TimeSeriesDataFile('initial_features_edited.csv', TimeSeriesDataFile.Formats.Alternate)

# Extract features
f_features  = f.ExtractAllFeatures()
f2_features = f2.ExtractAllFeatures()
