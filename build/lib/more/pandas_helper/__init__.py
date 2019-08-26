import pandas as pd
import warnings


@pd.api.extensions.register_dataframe_accessor("helper")
class pandas_helper(object):
    """
    TODO
    Value Counts df['col_name'].value_counts()
    NA Values
    Unique Values (nunique) df['col_names'].nunique()
    Null Values
      null_vals = df.isnull().sum(axis = 0)
      print(null_vals[null_vals != 0])
    Add options for Datatime, TimeDeltas, etc
    (check select_dtypes documentation)
    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self.cat_features = self._obj.select_dtypes(
            include=['object', 'category']).columns
        self.num_features = self._obj.select_dtypes(include=['number']).columns

    @staticmethod
    def _validate(obj):
        """
        Technically not needed for this since helper can only be used with
        Pandas Dataframe. Will keep it here for future
        """
        if isinstance(obj, pd.DataFrame):
            pass
        else:
            raise TypeError("Object must be a Pandas Dataframe")

    # @property

    def describe_categorical(self, verbose=False):
        """
        Prints ategorical variable summaries for a Pandas Dataframe
        Default behavior in pd.DataFrame.describe() is to print only
        numeric variables if the dataframe contains both numeric and
        categorical variables. This extension provides more flexibility
        """
        if (self._cat_exists()):
            self.__print_dashes(45)
            print("Summary Statictics for Categorical Variables")
            self.__print_dashes(45)
            print(self._obj[self.cat_features].describe())

            if (verbose):
                self.level_counts()
        else:
            warnings.warn("Data does not have any categorical columns")

    def describe_numeric(self):
        """
        Prints numeric variable summaries for a Pandas Dataframe
        Same as default behavior in pd.DataFrame.describe()
        """
        if (self._num_exists()):
            self.__print_dashes(40)
            print("Summary Statictics for Numeric Variables")
            self.__print_dashes(40)
            print(self._obj[self.num_features].describe())
        else:
            warnings.warn("Data does not have any numeric columns")

    def describe(self, verbose=False):
        """
        Prints both numeric and categorical variable summaries for a Pandas
        Dataframe. Default behavior in pd.DataFrame.describe() is to print
        only numeric variables if the dataframe contains both numeric and
        categorical variables. This extension provides more flexibility
        """
        self.describe_numeric()
        self.describe_categorical(verbose=verbose)

    def level_counts(self):
        """
        Printing number of observations in each level of categorical variables
        """
        self.__print_dashes(70)
        print("Printing number of observations in each level of "
              "categorical variables")
        self.__print_dashes(70)
        for feature in self.cat_features:
            print(feature)
            print(self._obj[feature].value_counts())

    def drop_columns(self, drops, inplace=False):
        """
        Extension of pd.DataFrame.drop(). Instead of giving an error on
        missing column name, this extension only gives a warnign and proceeds
        with execution
        """
        if (type(drops) == str):
            drops = [drops]  # Convert to list
        
        cols_to_del = []
        cols_not_present = []
        for drop in drops:
#            print(drop)
#            print(self._obj.columns)
            if drop in self._obj.columns:
                cols_to_del.append(drop)
            else:
                cols_not_present.append(drop)

        if (len(cols_not_present) > 0):
            warnings.warn(
                "Column(s) {} were not present in the dataframe".format(
                        cols_not_present))

        if (inplace):
            self._obj.drop(cols_to_del, axis=1, inplace=inplace)
            self.__set_all_feature_types()
        else:
            return(self._obj.drop(cols_to_del, axis=1, inplace=inplace))

    def map_columns(self, mapping):
        """
        mapping: Dictionary of column name mapping
        """
        self._obj.rename(columns=mapping, inplace=True)

    def add_columns(self, names, value=""):
        """
        previously called add_new_col
        adds a new column with a single value for all rows.
        names is a list of column names to be added
        """
        if (type(names) == str):
            names = [names]  # Convert to list
            
        for name in names:  # List of values
            self._obj[name] = value

    def filter_change(self, filter_col, filter_value,
                      change_col, change_value):
        """
        filter_col: Column name to filter by
        filter_value: value to filter
        change_col: Column to change values
        change_value: Value to change to in the changeCol column
        """
        self._obj.loc[self._obj[filter_col] == filter_value,
                      [change_col]] = change_value
                      
        return(self._obj)

    def filter_change_to_col(self, filter_col, filter_value,
                             change_col, change_value_colname):
        """
        filter_col: Column name to filter by
        filter_value: value to filter
        change_col: Column to change values
        change_value_colname: Column from where to pick new value
        """
        #self._obj = self._obj.apply(lambda row: self._filter_change_to_col(row),
        #    args=(filter_col, filter_value, change_col, change_value_colname),
        #    axis=1)
        
        self._obj = self._obj.apply(self._filter_change_to_col,
            args=(filter_col, filter_value, change_col, change_value_colname),
            axis=1)
        return(self._obj)
        

    def filter_delete(self, delete_col, delete_values):
        """
        deleteCol: Column to delete observations from
        deleteValue: If the value of the col_name in an observation (row) is
        equal to any value in col_values then that observation is deleted from
        the dataset.
        Returns the filtered dataframe (operation is not inplace so it will
        need to be assigned back from the call,
        e.g. df = df.helper.filter_delete(deleteCol, deleteValues))
        """
        if (type(delete_values) == str):
            delete_values = [delete_values]  # Convert to list

        keep = self._obj[delete_col].apply(
                self._absent,
                filter_list=delete_values)
        self._obj = self._obj[keep]
        return(self._obj[keep])
        
            
    def concat_columns(self, new_col_name, concat_list, concatBy=" "):
        self._obj[new_col_name] = self._obj[concat_list].apply(
                lambda row: concatBy.join(row.values.astype(str)), axis=1)
        
        return(self._obj)

    def strip_columns(self, names):
        self._obj[names] = self._obj[names].apply(
                lambda x: x.str.strip(), axis=0)

        return(self._obj)

    def title_case(self, names):
        # convert to string as some are treated as float
        self._obj[names] = self._obj[names].apply(
                lambda x: x.str.title(), axis=0)
        
        return(self._obj)

    def upper_case(self, names):
        # convert to string as some are treated as float
        self._obj[names] = self._obj[names].apply(
                lambda x: x.str.upper(), axis=0)
        
        return(self._obj)

    def to_datetime(self, names):
        if (type(names) == str):
            names = [names]  # Convert to list
        self._obj[names] = self._obj[names].apply(lambda x: pd.to_datetime(x), axis=0)
        
        return(self._obj)

    def select(self, names, inplace=False):
        if inplace:
            self._obj = self._obj[names]

        return(self._obj[names])

    ###################
    # Private Methods #
    ###################

    def __set_cat_features(self):
        self.cat_features = self._obj.select_dtypes(
            include=['object', 'category']).columns

    def __set_num_features(self):
        self.num_features = self._obj.select_dtypes(include=['number']).columns

    def __set_all_feature_types(self):
        self.__set_cat_features()
        self.__set_num_features()

    def _cat_exists(self):
        if (len(self.cat_features) > 0):
            return True
        else:
            return False

    def _num_exists(self):
        if (len(self.num_features) > 0):
            return True
        else:
            return False

    def __print_dashes(self, num=20):
        print("-"*num)

    def _present(self, value, filter_list):
        """
        Meant to be used with a Pandas DataFrame and apply function
        Returns a boolean Series indicating if the value is present in the
        filter_list or not.
        Since this is to be used with a pandas apply function, the "value" is
        a single row and column intersection value
        """
        if (value in filter_list):
            return True
        else:
            return False

    def _absent(self, value, filter_list):
        """
        Meant to be used with a Pandas DataFrame and apply function
        Returns a boolean Series indicating if the value is absent in the
        filter_list or not.
        Since this is to be used with a pandas apply function, the "value" is
        a single row and column intersection value
        """
        if (value in filter_list):
            return False
        else:
            return True
        
    def _filter_change_to_col(self, row, filter_col, filter_value,
                              change_col, change_value_colname):
        """
        filter_col: Column name to filter by
        filter_value: value to filter
        change_col: Column to change values
        change_value_colname: Column from where to pick new value
        """
        if row[filter_col] == filter_value:
            row[change_col] = row[change_value_colname]
        return row
