
def trim_spines(ax):
    for n in ['right','top']:
        ax.spines[n].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    return ax

def remove_spines(ax):
    for n in ['right','top','left','bottom']:
        ax.spines[n].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])    
    return ax

def set_spines_at_zero(ax):
    ax.spines['left'].set_position(('data', 0.))
    ax.spines['bottom'].set_position(('data', 0.))
    return ax


