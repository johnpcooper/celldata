import Tkinter,tkFileDialog
import numpy as np
import pandas as pd

class BaseDataSet(object):

    def __init__(self):
        return None
    
    def get_files(self):    
        # define lists of file names referring to the data
        # for each channel in your image
        root = Tkinter.Tk()
        self.dsred_files = tkFileDialog.askopenfilenames(parent=root, title='Choose dsred files')
        self.dsred_files_list = root.tk.splitlist(self.dsred_files)
        root.destroy()
        self.dsred_fns = sorted(self.dsred_files_list)
        
        root = Tkinter.Tk()
        self.yfp_files = tkFileDialog.askopenfilenames(parent=root, title='Choose yfp files')
        self.yfp_files_list = root.tk.splitlist(self.yfp_files)
        root.destroy()
        self.yfp_fns = sorted(self.yfp_files_list)
        
    def make_dfs(self, fns):
        # return a list of dfs made by reading the csv files referred to by the fns list
        # of filenames
        dfs = []
        for i in range(0, len(fns)):
            dfs.append(pd.read_csv(fns[i]))
        return dfs

    def calc_bg(self, df):
        # return a background value for an individual df
        # calculate total intensities of bg and cells        
        bg_I = df.iloc[-1]['RawIntDen']
        cells_I = df.iloc[0:-2]['RawIntDen'].sum()

        bg_A = df.iloc[-1]['Area']
        cells_A = df.iloc[0:-2]['Area'].sum()

        bg_final = (bg_I - cells_I) / (bg_A - cells_A)
        return bg_final

    def calc_bg_list(self, dfs):
        # return a list of background values for each df
        # in the list of dfs passed to the function
        bg_list = []
        
        for i in range(0, len(dfs)):
            bg_list.append(self.calc_bg(dfs[i]))
            
        return bg_list

    def add_bg_sub_col(self, df, bg):
        # add a column to the df passed to this function that contains
        # bg subtracted mean I values
        df['bg_sub_mean'] = df['Mean'] - bg

    def bg_subtract(self, dfs, bgs):
        # iterate add_bg_sub_col() over the list of dataframes
        for i in range(0, len(dfs)):
            self.add_bg_sub_col(dfs[i], bgs[i])
            
    def drop_bg(self, df):
        # return the input df with the last row removed
        df_pruned = df.drop(len(df) - 1)
        return df_pruned
    
    def drop_all_bgs(self, dfs):
        # return the input dfs with their last row removed by iterating
        # drop_bg() over the list of dfs passed this function
        pruned_dfs = []
        for i in range(0, len(dfs)):
            dropped = self.drop_bg(dfs[i])
            pruned_dfs.append(dropped)

        return pruned_dfs

class DataSet(BaseDataSet):

    def __init__(self):
        return None
                   
    def create_df(self):
        self.get_files()
        
        self.dsred_dfs = self.make_dfs(self.dsred_fns)
        self.yfp_dfs = self.make_dfs(self.yfp_fns)
        
        self.dsred_bgs = self.calc_bg_list(dsred_dfs)
        self.yfp_bgs = self.calc_bg_list(yfp_dfs)
        
        self.bg_subtract(self.dsred_dfs, self.red_bgs)
        self.bg_subtract(self.yfp_dfs, self.yfp_bgs)

        self.dsred_pruned_dfs = self.drop_all_bgs(self.dsred_dfs)
        self.yfp_pruned_dfs = self.drop_all_bgs(self.yfp_dfs)

        self.dsred_zipped_df = pd.concat(self.dsred_pruned_dfs, ignore_index=True)
        self.yfp_zipped_df = pd.concat(self.yfp_pruned_dfs, ignore_index=True)
        
        # Create a final df containing both dsred and yfp measurements
        df_final = self.dsred_zipped_df
        df_final['raw_dsred'] = self.dsred_zipped_df['RawIntDen']
        df_final['raw_yfp'] = self.yfp_zipped_df['RawIntDen']
        df_final['bg_sub_dsred'] = self.dsred_zipped_df['bg_sub_mean']
        df_final['bg_sub_yfp'] = self.yfp_zipped_df['bg_sub_mean']
        df_final['yfp/dsred'] = self.yfp_zipped_df['bg_sub_mean'] / self.dsred_zipped_df['bg_sub_mean']
        
        return df_final


class DataSetExpSeries(BaseDataSet):
    def __init__(self):
        self.exp_list = [100, 200, 400,
                         800, 1600, 3200]  
    
    def add_exp(self, dfs):
        for i in range(0, len(self.exp_list)):     
            dfs[i]['exposure'] = self.exp_list[i]
            
    def create_df(self):
        self.get_files()
        
        self.dsred_dfs = self.make_dfs(self.dsred_fns)      
        self.yfp_dfs = self.make_dfs(self.yfp_fns)
        
        self.dsred_bgs = self.calc_bg_list(self.dsred_dfs)
        self.yfp_bgs = self.calc_bg_list(self.yfp_dfs)
        
        self.bg_subtract(self.dsred_dfs, self.dsred_bgs)
        self.bg_subtract(self.yfp_dfs, self.yfp_bgs)

        self.dsred_pruned_dfs = self.drop_all_bgs(self.dsred_dfs)
        self.yfp_pruned_dfs = self.drop_all_bgs(self.yfp_dfs)
        
        self.add_exp(self.dsred_pruned_dfs)
        self.add_exp(self.yfp_pruned_dfs)

        self.dsred_zipped_df = pd.concat(self.dsred_pruned_dfs, ignore_index=True)
        self.yfp_zipped_df = pd.concat(self.yfp_pruned_dfs, ignore_index=True)
        
        # Create a final df containing both dsred and yfp measurements
        df_final = self.dsred_zipped_df
        df_final['raw_dsred'] = self.dsred_zipped_df['RawIntDen']
        df_final['raw_yfp'] = self.yfp_zipped_df['RawIntDen']
        df_final['bg_sub_dsred'] = self.dsred_zipped_df['bg_sub_mean']
        df_final['bg_sub_yfp'] = self.yfp_zipped_df['bg_sub_mean']
        df_final['yfp/dsred'] = self.yfp_zipped_df['bg_sub_mean'] / self.dsred_zipped_df['bg_sub_mean']
        
        return df_final
