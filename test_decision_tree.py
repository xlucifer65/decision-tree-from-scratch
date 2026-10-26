"""Run with: python -m unittest -v test_decision_tree.py"""
import unittest
import pandas as pd
from decision_tree_visualizer import entropy, information_gain, best_split, build_tree, predict_one, predict


class DecisionTreeTests(unittest.TestCase):
    def test_entropy(self):
        self.assertAlmostEqual(entropy(pd.Series(['Yes', 'No'])), 1.0)

    def test_numerical_threshold(self):
        df = pd.DataFrame({'Age': [18, 22, 25, 30, 35, 40], 'Buy': ['No']*3 + ['Yes']*3})
        self.assertEqual(best_split(df, 'Age', 'Buy')['threshold'], 27.5)
        self.assertAlmostEqual(information_gain(df, 'Age', 'Buy'), 1.0)
        self.assertEqual(predict_one(build_tree(df, 'Buy'), {'Age': 29}), 'Yes')

    def test_missing_numeric(self):
        df = pd.DataFrame({'Age': [20, 25, None, 35, 40], 'Play': ['No','No','No','Yes','Yes']})
        split = best_split(df, 'Age', 'Play')
        self.assertEqual(split['threshold'], 30.0)
        self.assertAlmostEqual(split['gain'], 0.9709505944546686)
        tree = build_tree(df, 'Play')
        self.assertEqual(predict_one(tree, {'Age': None}), 'No')
        self.assertEqual(sum(b['child']['samples'] for b in tree['branches']), len(df))

    def test_missing_category_does_not_collide_with_literal(self):
        df = pd.DataFrame({'Weather': [None, '<missing>', 'Sunny'], 'Play': ['No','Yes','Yes']})
        tree = build_tree(df, 'Play')
        self.assertEqual(predict_one(tree, {'Weather': None}), 'No')
        self.assertEqual(predict_one(tree, {'Weather': '<missing>'}), 'Yes')

    def test_mixed_types_are_distinct(self):
        df = pd.DataFrame({'Code': pd.Series([1, '1'], dtype=object), 'Target': ['No','Yes']})
        tree = build_tree(df, 'Target')
        self.assertEqual(predict_one(tree, {'Code': 1}), 'No')
        self.assertEqual(predict_one(tree, {'Code': '1'}), 'Yes')

    def test_missing_categorical(self):
        df = pd.DataFrame({'Weather': ['Sun', None, 'Rain'], 'Play': ['No','Yes','No']})
        self.assertEqual(predict_one(build_tree(df, 'Play'), {'Weather': None}), 'Yes')

    def test_constant_and_majority(self):
        df = pd.DataFrame({'X': [1,1,1], 'Y': ['Yes','No','Yes']})
        tree = build_tree(df, 'Y')
        self.assertTrue(tree['leaf'])
        self.assertEqual(tree['prediction'], 'Yes')

    def test_unseen_category_fallback(self):
        df = pd.DataFrame({'X': ['a','a','b'], 'Y': ['Yes','Yes','No']})
        self.assertEqual(predict_one(build_tree(df, 'Y'), {'X': 'unseen'}), 'Yes')

    def test_max_depth(self):
        df = pd.DataFrame({'X': ['a','b'], 'Y': ['Yes','No']})
        self.assertTrue(build_tree(df, 'Y', max_depth=0)['leaf'])

    def test_missing_target_rejected(self):
        with self.assertRaises(ValueError):
            build_tree(pd.DataFrame({'X':[1,2], 'Y':['Yes',None]}), 'Y')

    def test_training_predictions(self):
        df = pd.DataFrame({'Age':[18, 30, 40], 'Y':['No','Yes','Yes']})
        tree = build_tree(df, 'Y')
        self.assertEqual(predict(tree, df[['Age']]).tolist(), df['Y'].tolist())


if __name__ == '__main__':
    unittest.main()
