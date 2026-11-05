"""Exercise the notebook's actual split using synthetic multi-series rows."""
import json
from pathlib import Path
import unittest

import pandas as pd


class ChronologicalSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
                               'Demand Forecasting.ipynb').read_text())
        cls.split_source = next(
            ''.join(cell['source']) for cell in notebook['cells']
            if 'X_train, X_test, y_train, y_test = train_test_split' in ''.join(cell['source'])
            or 'validation_start = pd.Timestamp' in ''.join(cell['source'])
        )
        cls.fit_source = next(''.join(cell['source']) for cell in notebook['cells']
                              if 'model = reg.fit(X_train' in ''.join(cell['source']))

    def run_split(self, frame):
        namespace = dict(pd=pd, pdf=frame, FEATURES=['Store'], TARGET=['Weekly_Sales'])
        exec(self.split_source, namespace)
        return namespace

    def test_future_dates_and_all_series_are_held_out(self):
        dates = pd.to_datetime(['2012-01-27', '2012-02-01', '2012-04-30',
                                '2012-05-01', '2012-05-04'])
        frame = pd.DataFrame([
            dict(Date=date, Store=store, Weekly_Sales=100 * store + i)
            for i, date in enumerate(dates) for store in (1, 2)
        ]).set_index('Date').sample(frac=1, random_state=130)
        ns = self.run_split(frame)
        self.assertLess(ns['X_train'].index.max(), ns['X_test'].index.min())
        train, validation, test = [ns['X_' + name] for name in ('train', 'validation', 'test')]
        self.assertLess(train.index.max(), validation.index.min())
        self.assertLess(validation.index.max(), test.index.min())
        self.assertEqual(set(test.index), set(dates[3:]))
        self.assertEqual(len(train) + len(validation) + len(test), len(frame))
        for name in ('train', 'validation', 'test'):
            x, y = ns['X_' + name], ns['y_' + name]
            self.assertTrue(x.index.equals(y.index))
            self.assertTrue(x.index.is_monotonic_increasing)
            self.assertTrue((x.groupby(level=0).size() == 2).all())
        # Execute the real fitting cell with a spy, without importing XGBoost.
        class RegressorSpy:
            def __init__(self, **kwargs):
                pass

            def fit(self, x, y, *, eval_set, **kwargs):
                self.eval_set = eval_set
                return self

        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        with patch.dict(sys.modules, {'xgboost': SimpleNamespace(XGBRegressor=RegressorSpy)}):
            exec(self.fit_source, ns)
        self.assertIs(ns['reg'].eval_set[1][0], ns['X_validation'])
        self.assertIs(ns['reg'].eval_set[1][1], ns['y_validation'])

    def test_missing_period_is_rejected(self):
        frame = pd.DataFrame({'Store': [1], 'Weekly_Sales': [100]},
                             index=pd.to_datetime(['2012-05-04']))
        with self.assertRaises(ValueError):
            self.run_split(frame)


if __name__ == '__main__':
    unittest.main()
