"""Historical feature regression checks without TensorFlow or holiday downloads."""
import ast
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd


class FeatureHistoryTest(unittest.TestCase):
    def build(self, frame):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
                               'Demand Forecasting.ipynb').read_text())
        source = next(''.join(c['source']) for c in notebook['cells']
                      if 'def create_seasonal_features' in ''.join(c.get('source', [])))
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef)
                        and n.name == 'create_seasonal_features')
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        # Calendar is independent of sales and isolated from this regression.
        ns = dict(np=np, pd=pd, timedelta=timedelta,
                  holidays=SimpleNamespace(US=lambda: {}))
        exec(compile(module, '<notebook feature builder>', 'exec'), ns)
        return ns['create_seasonal_features'](frame)

    def frame(self):
        return pd.DataFrame({'Weekly_Sales': np.arange(1., 121.)},
                            index=pd.date_range('2020-01-03', periods=120, freq='7D'))

    def test_future_sales_cannot_change_earlier_features(self):
        original = self.frame()
        changed = original.copy()
        changed.iloc[80:, 0] += 100000
        cols = [c for c in self.build(original).columns if c.startswith(('Sales_', 'Month_Avg', 'Week_Avg'))]
        pd.testing.assert_frame_equal(self.build(original)[cols].iloc[:80],
                                      self.build(changed)[cols].iloc[:80])
        pd.testing.assert_frame_equal(self.build(original)[cols].iloc[:80],
                                      self.build(original.iloc[:80])[cols])

    def test_fallback_and_month_statistics_use_available_history(self):
        original = self.frame()
        result = self.build(original)
        self.assertEqual(result['Sales_Lag_52'].iloc[1], 1.5)
        for pos in (0, 20, 60, 110):
            prefix = original.iloc[:pos + 1]
            month = prefix[prefix.index.month == original.index[pos].month]
            expected = original.iloc[pos, 0] / month['Weekly_Sales'].mean()
            self.assertAlmostEqual(result['Month_Avg_Sales'].iloc[pos], expected)

    def test_unsorted_or_duplicate_dates_are_rejected(self):
        frame = self.frame()
        with self.assertRaises(ValueError):
            self.build(frame.iloc[::-1])
        with self.assertRaises(ValueError):
            self.build(pd.concat([frame.iloc[:1], frame]))


if __name__ == '__main__':
    unittest.main()
