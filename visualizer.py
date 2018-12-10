from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
import random


def load_data(folder, limit=24):
    # Model 1
    mod1 = []
    f = open(folder + '/model1results', 'r')
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
    r = lambda: random.randint(0, 255)

    print("Loading data...")
    mod1, mod2 = load_data('./results')

    print("Creating visuals...")
    output_file('results.html', title="Twitter data")

    # Mod 1 good
    print(mod1[0][0])
    s1 = figure(x_range=mod1[0][0].split(' '), width=450, plot_height=250, title='Model 1 Good (Last Update:' + str(mod1[0][-1]) + ')', toolbar_location=None, tools="")
    s1.yaxis.axis_label = 'Count'
    s1.vbar(x=mod1[0][0].split(' '), top=mod1[0][1].split(' '), width=0.9, color='#%02X%02X%02X' % (r(), r(), r()))
    # Mod 1 neutral
    s2 = figure(x_range=mod1[0][0].split(' '), width=450, plot_height=250, title='Model 1 Neutral(Last Update:' + str(mod1[0][-1]) + ')', toolbar_location=None, tools="")
    s2.yaxis.axis_label = 'Count'
    s2.vbar(x=mod1[0][0].split(' '), top=mod1[0][2].split(' '), width=0.9, color='#%02X%02X%02X' % (r(), r(), r()))
    # Mod 1 bad
    s3 = figure(x_range=mod1[0][0].split(' '), width=450, plot_height=250, title='Model 1 Bad (Last Update:' + str(mod1[0][-1]) + ')', toolbar_location=None, tools="")
    s3.yaxis.axis_label = 'Count'
    s3.vbar(x=mod1[0][0].split(' '), top=mod1[0][3].split(' '), width=0.9, color='#%02X%02X%02X' % (r(), r(), r()))

    # Mod 2 good
    s4 = figure(x_range=mod2[0][0].split(' '), width=450, plot_height=250, title='Model 2 Good (Last Update:' + str(mod2[0][-1]) + ')', toolbar_location=None, tools="")
    s4.yaxis.axis_label = 'Count'
    s4.vbar(x=mod2[0][0].split(' '), top=mod2[0][1].split(' '), width=0.9, color='#%02X%02X%02X' % (r(), r(), r()))
    # Mod 2 neutral
    s5 = figure(x_range=mod2[0][0].split(' '), width=450, plot_height=250, title='Model 2 Neutral(Last Update:' + str(mod2[0][-1]) + ')', toolbar_location=None, tools="")
    s5.yaxis.axis_label = 'Count'
    s5.vbar(x=mod2[0][0].split(' '), top=mod2[0][2].split(' '), width=0.9, color='#%02X%02X%02X' % (r(), r(), r()))
    # Mod 2 bad
    s6 = figure(x_range=mod2[0][0].split(' '), width=450, plot_height=250, title='Model 2 Bad (Last Update:' + str(mod2[0][-1]) + ')', toolbar_location=None, tools="")
    s6.yaxis.axis_label = 'Count'
    s6.vbar(x=mod2[0][0].split(' '), top=mod2[0][3].split(' '), width=0.9, color='#%02X%02X%02X' % (r(), r(), r()))

    p = gridplot([[s1, s2, s3], [s4, s5, s6]])

    # show the results
    print('Done')
    if visualize:
        show(p)