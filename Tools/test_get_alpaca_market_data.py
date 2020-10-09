from get_alpaca_market_data import market_data
import pytest
import datetime

md = market_data()
endDate = "2020-10-02T00:00:00Z"

def test_getPrice():
    price = md.getPrice('1Min', 'SPY', 1, endDate)
    assert isinstance(price[0].c, (int, float))
    assert isinstance(price[0].h, (int, float))
    assert isinstance(price[0].l, (int, float))
    assert isinstance(price[0].o, (int, float))
    assert isinstance(price[0].t, (datetime.datetime))
    assert isinstance(price[0].v, (int, float))