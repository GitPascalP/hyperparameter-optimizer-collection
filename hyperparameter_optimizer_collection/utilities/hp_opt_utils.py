"""

Different function or classes, that help and can make things easier

"""
import json
import datetime
import os
import matplotlib.pyplot as plt


def set_plot_params(figure_title=22, figsize=(20, 14), title_size=24,
                    label_size=20, tick_size=16, marker_size=10, line_width=3,
                    legend_font=16, style='dark'):
    """ setting matplotlib params """

    if style == 'dark':
        plt.style.use('dark_background')
    else:
        plt.style.use(style)

    params = {'axes.titlesize': title_size,
              'legend.fontsize': legend_font,
              'figure.figsize': figsize,
              'axes.labelsize': label_size,
              'xtick.labelsize': label_size,
              'ytick.labelsize': label_size,
              'figure.titlesize': figure_title}
    plt.rcParams.update(params)


# saving and loading jsons
def save_to_json(data, name, indent=4, sort=False, ending='.json'):
    """saving dictionary to json file"""
    with open(name + ending, 'w') as json_file:
        json.dump(data, json_file, indent=indent, sort_keys=sort)
    print('Done writing')


def load_json(filepath, name, ending='.json'):
    """loading json file and return it as dict"""
    with open(filepath + name + ending, "r") as f:
        data_json = f.read()

    data = json.loads(data_json)
    return data


def go_dir_up(levels_up=1):
    """  get an upper directory """
    current_dir = os.getcwd()
    dir_list = current_dir.split('/')

    for i in range(levels_up):
        dir_list.pop()
    upper_dir = '/'.join(dir_list)
    return upper_dir + '/'


class SavingResults:
    """

    """
    def __init__(self, algorithm, name, up=1):
        """
        args:
            algorithm(str): used algorithm

        """
        self.root_dir = go_dir_up(levels_up=up)
        self.algorithm = algorithm
        self.name = name

    def generate_id(self,):
        """ generate id for save-folder """
        return self.algorithm + datetime.date() + '/'

    def save_results(self, data, data_type='results'):
        """
        args:
            data(dict): data to store
            data_type: 'results' or 'config'
        """
        # path = self.root_dir + 'saves/' + self.generate_id() + data_type + '/'
        path = self.root_dir + 'saves/' + data_type + '/'
        #save_to_json(data, path + self.name)
        with open(path + self.name + '.txt', "a") as f:
            f.write(json.dumps(data) + "\n")
