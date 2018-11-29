import ccxt
import time

class BtcEcc(object):
    @staticmethod
    def get_exchanges():
        return ccxt.exchanges

    @staticmethod
    def get_exchange(exchange_id, apiKey, secret):
        #exchange_id = 'huobi'
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'apiKey': apiKey,
            'secret': secret,
            'timeout': 30000,
            'enableRateLimit': True,
        })
        return exchange

    @staticmethod
    def get_order_books(exchange, symbol, limit):
        return exchange.fetch_order_book(symbol, limit)

    @staticmethod
    def get_best_bid(order_books):
        bid_item = order_books['bids'][0]
        return bid_item[0], bid_item[1]

    @staticmethod 
    def get_best_ask(order_books):
        ask_item = order_books['asks'][0]
        return ask_item[0], ask_item[1]

    @staticmethod
    def get_tickers(exchange, symbol):
        '''
        获取指定币对的价格信息
        '''
        bid_file = open('bid.txt', 'w', encoding='utf8')
        ask_file = open('ask.txt', 'w', encoding='utf8')
        bid_prev_item = None
        ask_prev_item = None
        idx = 0
        for i in range(5):
            ticker = exchange.fetch_ticker(symbol)
            bid_prev_item = BtcEcc.fill_price_item(bid_file, ticker, bid_prev_item, idx, 'bid')
            ask_prev_item = BtcEcc.fill_price_item(ask_file, ticker, ask_prev_item, idx, 'ask')
            idx += 1
            time.sleep(0.1)
        bid_file.close()
        ask_file.close()

    @staticmethod
    def fill_price_item(sample_file, ticker, prev_item, idx, bidOrAsk):
        item = []
        item.append(ticker['high'])
        item.append(ticker['low'])
        item.append(ticker['bid'])
        item.append(ticker['bidVolume'])
        item.append(ticker['ask'])
        item.append(ticker['askVolume'])
        item.append(ticker['open'])
        item.append(ticker['close'])
        item.append(ticker['change'])
        item.append(ticker['percentage'])
        item.append(ticker['baseVolume'])
        item.append(ticker['quoteVolume'])
        item.append(-1.0)
        if idx > 0:
            prev_item[len(prev_item)-1] = ticker[bidOrAsk]
            sample_file.write(','.join(list(map(str, prev_item))) + '\r\n')
        return item

    @staticmethod
    def convert_item_to_str(item):
        rec = ''
        rec += str(item[0])
        for idx in range(len(item)-1):
            rec += ',' + str(item[idx+1])
        rec += '\r\n'
        return rec

def test1():
    bte = 'BTC/USDT'
    etb = 'USDT/BTC'
    symbol = bte
    exchange = BtcEcc.get_exchange('huobipro', '356cda29-253918bb-573a5afa-eeeae', 'a80d3cfd-4db6c763-0b73daee-cba31')
    print(exchange)
    markets = exchange.load_markets()
    # print('type:{0} {1}!'.format(type(markets), markets))
    limit = 10
    #obs = exchange.fetch_order_book('ETH/BTC', limit)
    order_books = BtcEcc.get_order_books(exchange, symbol, limit)
    bid_price, bid_amount = BtcEcc.get_best_bid(order_books)
    ask_price, ask_amount = BtcEcc.get_best_ask(order_books)
    print('bids:{0} --- {1}!'.format(bid_price, bid_amount))
    print('asks:{0} --- {1}!'.format(ask_price, ask_amount))
    print('delta:{0}!'.format(bid_price-ask_price))
    '''
    if exchange.has['fetchTicker']:
        print(exchange.fetch_ticker('ETH/BTC'))
    if exchange.has['fetchTrades']:
        print(exchange.fetch_trades('ETH/BTC'))
    '''
    #print('balance:{0}!'.format(exchange.fetch_balance()))
    # sell BTC and buy USDT
    #sell_btc_order = exchange.create_limit_sell_order(symbol, 0.001, ask_price)
    #print('sell order: {0}!'.format(sell_btc_order))
    buy_btc_order = exchange.create_limit_buy_order(symbol, 0.001, bid_price)
    print('buy_order: {0}!'.format(buy_btc_order))

def test2():
    bte = 'BTC/USDT'
    symbol = bte
    exchange = BtcEcc.get_exchange('huobipro', '356cda29-253918bb-573a5afa-eeeae', 'a80d3cfd-4db6c763-0b73daee-cba31')
    BtcEcc.get_tickers(exchange, symbol)


def main():
    #test1()
    test2()
    with open('a1.txt', 'a', encoding='utf8') as fd:
        fd.write('msg5:Hello world\r\n')
    print('bye!')

if '__main__' == __name__:
    main()
