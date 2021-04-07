import os
import glob
import json


def load_json_files(folder_name, sort_by_size):
        """Load molecules from json files
            
            folder_name:  String containing the initial molecule directory.
            sort_by_size: Boolean which specifies if molecules are sorted by size.

            returns: Sorted list with molecule files.

        """
        all_files = glob.glob(f'{folder_name}*.'+ self.file_ext)
        # if '/' in self.folder_name:
        #     self.folder_name = self.folder_name.split('/')[0]

        # self.in_order = in_order
        if sort_by_size:
            all_files.sort(key=os.path.getsize)
        else:
            all_files.sort()

        return all_files