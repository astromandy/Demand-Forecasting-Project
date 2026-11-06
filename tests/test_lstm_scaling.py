"""Run the notebook's preprocessing with synthetic features, without TensorFlow."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class LSTMScalingTest(unittest.TestCase):
    def preprocess(self, frame):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
                               'Demand Forecasting.ipynb').read_text())
        source = next(''.join(c['source']) for c in notebook['cells']
                      if 'def improved_sales_forecasting(' in ''.join(c['source']))
        workflow = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef)
                        and n.name == 'improved_sales_forecasting')
        # Execute the actual workflow up to model construction. Feature engineering
        # is deliberately isolated: this regression covers scaler leakage only.
        stop = next(i for i, n in enumerate(workflow.body)
                    if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'model' for t in n.targets))
        body = workflow.body[:stop] + [ast.parse('return locals()').body[0]]
        workflow.body = body
        module = ast.fix_missing_locations(ast.Module(body=[workflow], type_ignores=[]))
        ns = dict(np=np, pd=pd, RobustScaler=RobustScaler,
                  create_seasonal_features=lambda df: df.copy())
        exec(compile(module, '<notebook preprocessing>', 'exec'), ns)
        return ns['improved_sales_forecasting'](frame)

    def test_held_out_values_do_not_change_training_scale_or_sequences(self):
        frame = pd.DataFrame({'Weekly_Sales': np.arange(60, dtype=float)},
                             index=pd.date_range('2020-01-03', periods=60, freq='7D'))
        original = self.preprocess(frame)
        # 36 targets, 28 for training: first held-out target is row 24 + 28.
        boundary = 24 + int((len(frame) - 24) * 0.8)
        changed = frame.copy()
        changed.iloc[boundary:, 0] += 10000
        perturbed = self.preprocess(changed)
        np.testing.assert_allclose(original['scaler'].center_, perturbed['scaler'].center_)
        np.testing.assert_allclose(original['scaler'].scale_, perturbed['scaler'].scale_)
        np.testing.assert_allclose(original['X_train'], perturbed['X_train'])
        np.testing.assert_allclose(original['y_train'], perturbed['y_train'])
        expected = RobustScaler().fit(frame.iloc[:boundary])
        np.testing.assert_allclose(original['scaler'].center_, expected.center_)
        np.testing.assert_allclose(original['scaler'].scale_, expected.scale_)
        self.assertEqual(original['X_test'].shape, (8, 24, 1))
        restored = original['scaler'].inverse_transform(original['y_test'].reshape(-1, 1))
        np.testing.assert_allclose(restored[:, 0], frame.iloc[boundary:, 0])
        self.assertFalse(np.allclose(original['y_test'], perturbed['y_test']))


if __name__ == '__main__':
    unittest.main()
