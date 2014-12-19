

class Trader:
  class TradeResult:
    Buy     = 'Buy'
    Sell    = 'Sell'
    Nothing = 'Nothing'

  def __init__(self, predicted, actual, initial_amount = 10):
    self._predicted = predicted
    self._actual    = actual
    self._currentAmount = initial_amount

  def _AttemptTrade(self, i):
    predictions = self._predicted[i]
    actual      = self._actual[i - 1][0]
    if predictions[0] < actual and predictions[1] < predictions[0] and predictions[2] < predictions[1]:
      # We see that generally the trend is strongly negative, so we think we should sell
      return Trader.TradeResult.Sell
    elif predictions[0] > actual and predictions[1] > predictions[0] and predictions[2] > predictions[1]:
      # We see that generally the trend is strongly positive, so we think we should buy
      return Trader.TradeResult.Buy

    # Otherwise do nothing
    return Trader.TradeResult.Nothing

  def GenTrade(self):
    good_trades = 0
    bad_trades  = 0
    no_actions  = 0
    for i in range(1, len(self._predicted) - 1):

      trade_result = self._AttemptTrade(i)
      
      today_actual    = self._actual[i]
      tomorrow_actual = self._actual[i + 1]
      if trade_result == Trader.TradeResult.Buy:
        # If we bought, we want to ensure that the value of the price goes up
        if today_actual < tomorrow_actual:
          # The value did go up
          good_trades += 1
          yield True
        
        elif today_actual > tomorrow_actual:
          # The value went down. Lost on trade
          bad_trades += 1
          yield False
      elif trade_result == Trader.TradeResult.Sell:
        # If we sold, we want to ensure value of price goes down
        if today_actual < tomorrow_actual:
          # We sold even though the value went up
          bad_trades += 1
          yield False
        elif today_actual > tomorrow_actual:
          # Sold and price went down
          good_trades += 1
          yield True
      else:
        no_actions += 1

      i += 1

    yield good_trades, bad_trades, no_actions
