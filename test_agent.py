import unittest
from datetime import datetime, timezone, timedelta
from agent import analyse, demo_snapshot

class QualityChecks(unittest.TestCase):
    def test_old_data_never_emits_buy(self):
        item=demo_snapshot()['stocks'][0]
        result=analyse(item,datetime.now(timezone.utc)+timedelta(days=3))
        self.assertEqual(result['trading'],'INSUFFICIENT DATA')

    def test_future_bars_are_ignored(self):
        item=demo_snapshot()['stocks'][0]
        result=analyse(item,datetime.now(timezone.utc)-timedelta(days=10))
        self.assertEqual(result['trading'],'INSUFFICIENT DATA')

    def test_invalid_ohlc_blocks_signal(self):
        item=demo_snapshot()['stocks'][0]; item['bars'][-1]['high']=0
        self.assertEqual(analyse(item,datetime.now(timezone.utc))['trading'],'INSUFFICIENT DATA')

    def test_no_fundamental_claim_from_chart(self):
        result=analyse(demo_snapshot()['stocks'][0],datetime.now(timezone.utc))
        self.assertEqual(result['investment'],'INSUFFICIENT DATA')

    def test_current_valid_bars_produce_analysis(self):
        result=analyse(demo_snapshot()['stocks'][0],datetime.now(timezone.utc))
        self.assertIn('support',result)
        self.assertIn(result['trading'],['BUY','SELL','HOLD / WAIT'])

if __name__=='__main__':
    unittest.main()
