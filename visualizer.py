from math import pi

import pandas as pd

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.transform import cumsum


def load_data(folder, limit=24):
    # Model 1
    mod1 = []
    f = open(folder+'/model1results', 'r')
    lines = f.readlines()
    count = 0
    for line in reversed(lines):
        if count > limit:  # Number of data points for graph
            break
        count += 1
        mod1.append(line.split(', '))
    f.close()

    # Model 2
    mod2 = []
    f = open(folder+'/model2results', 'r')
    lines = f.readlines()
    count = 0
    for line in reversed(lines):
        if count > limit:  # Number of data points for graph
            break
        count += 1
        mod2.append(line.split(', '))
    f.close()

    return mod1, mod2


def run(visualize=True):
    print('Loading data...')
    m1, m2 = load_data('./results')

    # Organizing data
    words = m1[-1][0].split(' ')
    m1_results = {}
    m2_results = {}
    for word in range(len(words)):
        m1_results[words[word]] = [m1[-1][1].split(' ')[word], m1[-1][2].split(' ')[word], m1[-1][3].split(' ')[word]] #Load as good, indiferent, bad.
        m2_results[words[word]] = [m2[-1][1].split(' ')[word], m2[-1][2].split(' ')[word], m2[-1][3].split(' ')[word]] #Load as good, indiferent, bad.

    print('Creating Visuals...')

    output_file("visuals.html")

    for word in m1_results:
        temp_dict

    for word in m2_results:
        print(word)

  #  data1 = pd.Series(m1_results).reset_index(name='value').rename(columns={'index':'good','bad'})
  #  data1['angle'] = data['value']/data['value'].sum() * 2*pi
   # data1['color'] = ['#7ca1e4', '#d8db9a', '#e00812']

    #p = figure(plot_height=350, title="graphthingTODOOD", toolbar_location=None,
     #          tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

   # p.wedge(x=0, y=1, radius=0.4,
    #        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
    #        line_color="white", fill_color='color', legend='sentiment', source=data)

   # p.axis.axis_label = None
   # p.axis.visible = False
    #p.grid.grid_line_color = None


    # put all the plots in a grid layout
   # p = gridplot([model1_figures, model2_figures])

    #if visualize:
      #  show(p)

run()