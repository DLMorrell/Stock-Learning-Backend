import alpaca_backtrader_api
import backtrader as bt
from datetime import datetime
from dotenv import load_dotenv
import os, csv

load_dotenv()

ALPACA_API_KEY = os.getenv('API')
ALPACA_SECRET_KEY = os.getenv('SECRET')
ALPACA_PAPER = True

file_name = "SPY.csv"

class SmaCross(bt.SignalStrategy):
  def __init__(self):
    sma1, sma2 = bt.ind.SMA(period=10), bt.ind.SMA(period=30)
    crossover = bt.ind.CrossOver(sma1, sma2)
    self.signal_add(bt.SIGNAL_LONG, crossover)


cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)

store = alpaca_backtrader_api.AlpacaStore(
    key_id=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    paper=ALPACA_PAPER
)

if not ALPACA_PAPER:
  broker = store.getbroker()  # or just alpaca_backtrader_api.AlpacaBroker()
  cerebro.setbroker(broker)

DataFactory = store.getdata  # or use alpaca_backtrader_api.AlpacaData
data0 = DataFactory(dataname='SPY', historical=True, fromdate=datetime(
    2015, 1, 1), timeframe=bt.TimeFrame.Days)

#writer = csv.writer(open(file_name, 'w'))
#writer.writerows(data0)
print(data0.bar)
for index, element in enumerate(data0.lines):
    
    print(f'{index} : {element}')

cerebro.adddata(data0)

#print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
#cerebro.run()
#print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
#cerebro.plot()