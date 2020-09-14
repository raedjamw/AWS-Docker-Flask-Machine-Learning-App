import unittest
import pandas as pd


class test_Transformations(unittest.TestCase):
    def test_dummies_funct(self):

        """
          Test that the series of categorical variables is properly converted to binary dataframe .
          The input is a actual dataframes and series used in the form of pkl files

        """
        from transformation_functions import dummies_funct
        input_3 = pd.read_pickle(r"test_3_input.pkl")
        output_3 = pd.read_pickle(r"test_3_output.pkl")
        # assertion statement
        actual = dummies_funct(input_3, input_3.name)

        self.assertTrue(actual.equals(output_3))

    def test_imputer_funct(self):

        """
        Test imputer function converts column nans to means
        The input is a actual dataframes and series used in the form of pkl files
        
        """
        from transformation_functions import imputer_func
        input_5 = pd.read_pickle("test_5_input.pkl")
        output_5 = pd.read_pickle(r"test_5_output.pkl")

        # assertion statement
        actual = imputer_func(input_5, 'mean', ['y', 'x5', 'x31',  'x81' ,'x82'])

        self.assertTrue(actual.equals(output_5))


# Programmatically building up the test_Transformations TestSuite
run_tests = unittest.TestLoader().loadTestsFromTestCase(test_Transformations)
# call the TestRunner with the verbosity 2
unittest.TextTestRunner(verbosity=2).run(run_tests)


