import glob
import os, pickle
from pytid.utils.configuration import Configuration


DATETIME_FMT_PRINTABLE = '%Y-%m-%d - %H:%M:%S'
conf = Configuration()
workfold = conf.gnss.get("working_dir")
os.makedirs(os.path.join(workfold, 'saved_data'), exist_ok=True)
saved_data_fold = os.path.join(workfold, 'saved_data')

def find_shared_objects(prefix: str) -> str:
    """
    Finds compiled C libraries available as shared objects under the build folder.
    :param prefix: The filename prefix for which you want to import
    :return: The relative path to the library.
    """
    found_paths = glob.glob(os.path.join("build", "**", f"{prefix}*"), recursive=True)
    if len(found_paths) == 0:
        raise ImportError("I could not find the shared object file. Have you run setup.py?")
    elif len(found_paths) > 1:
        raise ImportError(f"I found too many shared objects with prefix={prefix}. Found files: {found_paths}")
    return found_paths[0]

def load_data_from_pickle_file(file_path):
    with open(file_path, 'rb') as dat_f:
        dat = pickle.load(dat_f)
    return dat

def save_data_to_pickle_file(dat, file_path):
    with open(file_path, 'wb') as dat_f:
        pickle.dump(dat, dat_f)

def get_list_from_file(filepath):
    '''
    Converts a text file to a python list of strings, with each line becoming one item
    in the list.
    :param filepath:
    :return:
    '''
    myf=open(filepath,'r')
    ol=[]
    for i in myf:
        if i.strip()!='':
            ol.append(i.strip())
    myf.close()
    return ol

def write_list_to_file(mylist=None,filepath=None):
    '''
    Writes a list object to a file, one item per line.
    :param mylist: python list object
    :param filepath: file path for output
    :return:
    '''
    if mylist is None or filepath is None:
        print("usage: write_list_to_file(list,filepath)")
    myf=open(filepath,'w')
    for i in mylist:
        myf.write(str(i) + '\n')
    myf.close()

def write_dict_to_file(mydict, filepath, delimiter='\t'):
    '''
    Takes a dictionary object and writes it to a file with each entry one on line, in string form,
    separated by a delimiter.
    :param mydict: a dictionary object where both keys and values can be readily converted to srings
    :param filepath: path to the output text file
    :param delimiter: (self-explanatory, default is '\t')
    :return: None
    '''
    # if delimiter is None:
    #     delimiter = '\t'
    myf = open(filepath,'w')
    for k in mydict.keys():
        myf.write(k)
        myf.write(delimiter)
        myf.write(str(mydict[k]))
        myf.write("\n")
    myf.close()

def get_dict_from_file(filename, key_col=0, val_col=1, delimiter='\t', header_rows=0):
    '''
    General file to pull a dictionary object from a delimited text file. Default is to assume two columns,
    tab-delimited with keys in the first column and values in the second, with no header. But any column
    indices, delimiter and number of header rows can be specified.
    :param filename: path to the target file
    :param key_col: column index (0-indexed) to get the keys from
    :param val_col: column index (0-indexed) to get the values from
    :param delimiter: delimiter to split the columns by (no text qualifier, sorry)
    :param header_rows: number of rows at the start of the file to skip
    :return:
    '''
    myf = open(filename, 'r')
    for r in range(header_rows):
        foo = myf.readline()
    args={}
    ct = 0
    for ln in myf:
        if len(ln.strip()) > 0:
            a = ln.strip().split(delimiter)
            args[a[key_col]] = a[val_col]
            ct += 1
            if ct % 100000 == 0:
                print('\r',end=''); print('%15s lines done' % f'{ct:,d}', end = '')
    print('\n')
    myf.close()
    return args

#
# def get_full_station_list():
#     station_list_fp = conf.gnss.get("full_station_list")
#     return get_list_from_file(station_list_fp)
