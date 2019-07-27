import pandas as pd
import warnings

@pd.api.extensions.register_dataframe_accessor("helper")
class pandas_helper(object):
    """
    # TODO
    # Value Counts df['col_name'].value_counts()
    # NA Values
    # Unique Values (nunique) df['col_names'].nunique()
    # Null Values 
    #   null_vals = df.isnull().sum(axis = 0)
    #   print(null_vals[null_vals != 0])
    # Add options for Datatime, TimeDeltas, etc (check select_dtypes documentation)
    """
    
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self.cat_features = self._obj.select_dtypes(include=['object','category']).columns
        self.num_features = self._obj.select_dtypes(include=['number']).columns

    @staticmethod
    def _validate(obj):
        """
        Technically not needed for this since helper can only be used with Pandas Dataframe
        Will keep it here for future
        """
        if isinstance(obj, pd.DataFrame):
            pass
        else:
            raise TypeError("Object must be a Pandas Dataframe")
            

    #@property
    
    def describe_categorical(self, verbose=False):
        """
        Prints ategorical variable summaries for a Pandas Dataframe
        Default behavior in pd.DataFrame.describe() is to print only numeric variables if 
        the dataframe contains both numeric and categorical variables. 
        This extension provides more flexibility
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
    
    def describe(self,verbose=False):
        """
        Prints both numeric and categorical variable summaries for a Pandas Dataframe
        Default behavior in pd.DataFrame.describe() is to print only numeric variables if 
        the dataframe contains both numeric and categorical variables. 
        This extension provides more flexibility
        """
        self.describe_numeric()
        self.describe_categorical(verbose=verbose)
        
    def level_counts(self):
        """
        Printing number of observations in each level of categorical variables
        """
        self.__print_dashes(70)
        print("Printing number of observations in each level of categorical variables")
        self.__print_dashes(70)
        for feature in self.cat_features:
            print(feature)
            print(self._obj[feature].value_counts())
            
    def drop_columns(self, drops, inplace=False):
        """
        Extension of pd.DataFrame.drop(). Instead of giving an error on missing column name,
        this extension only gives a warnign and proceeds with execution
        """
        cols_to_del = []
        cols_not_present = []
        for drop in drops:
            if drop in self._obj.columns:
                cols_to_del.append(drop)
            else:
                cols_not_present.append(drop)
        
        if (len(cols_not_present) > 0):
            warnings.warn("Column(s) {} were not present in the dataframe".format(cols_not_present))
        
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
                
    def add_columns(self,names,value = ""):
        """
        previously called add_new_col
        adds a new column with a single value for all rows.
        names is a list of column names to be added         
        """
        if (type(names) == str):
            self._obj[names] = value
        else:
            for name in names:
                self._obj[name] = value
            
    def filter_change(self,filterCol,filterValue,changeCol,changeValue):
        """
        filterCol: Column name to filter by
        filterValue: value to filter
        changeCol: Column to change values
        changeValue: Value to change to in the changeCol column 
        """
        self._obj.loc[self._obj[filterCol] == filterValue, [changeCol]] = changeValue
        
    def filter_delete(self,deleteCol,deleteValue):
        self._obj = self._obj[self.data[deleteCol] != deleteValue]
            
    def concat_columns(self,newColName,column1,column2,concatBy = " "):
        self._obj[newColName] = self._obj[column1] + concatBy + self._obj[column2] 
        
    def strip_columns(self,names):
        self._obj[names] = self._obj[names].astype(str).map(lambda x: x.strip()) # convert to string as some are treated as float
        
    def title_case(self,names):
        self._obj[names] = self._obj[names].astype(str).map(lambda x: x.title()) # convert to string as some are treated as float
        
    def upper_case(self,names):
        self._obj[names] = self._obj[names].astype(str).map(lambda x: x.upper()) # convert to string as some are treated as float
        
    def select(self,names, inplace=False):
        if inplace:
            self._obj = self._obj[names]
            
        return(self._obj[names])
    
    #########################
    #### Private Methods ####
    #########################
    
    def __set_cat_features(self):
        self.cat_features = self._obj.select_dtypes(include=['object','category']).columns
        
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
    
    def __print_dashes(self,num = 20):
        print("-"*num)    
        
    
        