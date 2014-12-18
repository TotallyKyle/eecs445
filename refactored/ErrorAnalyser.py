import matplotlib.pyplot as plot

class ErrorAnalyser:

  def __init__(self, predicted, actual):
    self._predicted = predicted
    self._actual    = actual

  def AverageErrorForOutputs(self, outputs):
    return [self.AverageErrorForOutput(output) for output in outputs]

  def AverageErrorForOutput(self, output):
    total = 0
    for idx, predicted_row in enumerate(self._predicted):
      total += abs(predicted_row[output] - self._actual[idx][output])
    return total / len(self._predicted)

  def AddOutputErrorHistogram(self, output):
    print outputs

  def GenerateLineGraph(self):
    plot.plot(self._predicted)
    plot.show()
