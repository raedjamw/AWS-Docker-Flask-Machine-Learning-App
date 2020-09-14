import unittest
import pandas as pd


class test_Dataframes(unittest.TestCase):

    def test_df_from_csv(self):
        """

        check if a csv is read in as a DF

        """
        from dataframe_shape_functions import df_from_csv
        expected = pd.read_pickle(r"csv_to_df.pkl")
        actual = df_from_csv(r'csv_to_df.csv')
        self.assertTrue(actual.equals(expected))



    def test_series_df_concat(self):
        """
          Test that the  series/dataframes are concat along columns.
          The input is actual dataframes and series used in the form of pkl files

        """
        from dataframe_shape_functions import series_df_concat
        x_test = pd.read_pickle(r"test_1_x.pkl")
        y_test = pd.read_pickle(r"test_1_y.pkl")
        x_y_test = pd.read_pickle(r"test_1_result.pkl")
        # assertion statement
        actual = series_df_concat(x_test, y_test, True)

        self.assertTrue(actual.equals(x_y_test))

    def test_df_df_concat(self):
        from dataframe_shape_functions import df_df_concat
        """
          Test that the dataframes/dataframes are concat along columns.
          The input is a sample of the dataframes used in the form of pkl files

        """
        x_test = pd.read_pickle(r"test_2_x.pkl")
        y_test = pd.read_pickle(r"test_2_y.pkl")
        x_y_test = pd.read_pickle(r"test_2_result.pkl")
        # assertion statement
        actual = df_df_concat(x_test, y_test)

        self.assertTrue(actual.equals(x_y_test))



    def test_df_df_row_concat(self):
        from dataframe_shape_functions import df_df_row_concat
        """
            Concat dataframes along rows
            The input is a sample of the dataframes used in the form of pkl files

        """
        x_test = pd.read_pickle(r"test_4_x1.pkl")
        y_test = pd.read_pickle(r"test_4_x2.pkl")
        x_y_test = pd.read_pickle(r"test_4_result.pkl")
        # assertion statement
        actual = df_df_row_concat(x_test, y_test)

        self.assertTrue(actual.equals(x_y_test))


# Programmatically building up the TestSuite from the test_Dataframes class
run_tests = unittest.TestLoader().loadTestsFromTestCase(test_Dataframes)
# call the TestRunner with the verbosity 2
unittest.TextTestRunner(verbosity=2).run(run_tests)



