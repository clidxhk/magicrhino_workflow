from matplotlib.colors import LinearSegmentedColormap

def num_samples(df):
    num_samples = df.shape[0]
    return num_samples

def shap_cluster_color():
    cdict1 = { 
        "red": (
            (0.0, 0.11764705882352941, 0.11764705882352941),
            (1.0, 0.9607843137254902, 0.9607843137254902),
        ),
        "green": (
            (0.0, 0.5333333333333333, 0.5333333333333333),
            (1.0, 0.15294117647058825, 0.15294117647058825),
        ),
        "blue": (
            (0.0, 0.8980392156862745, 0.8980392156862745),
            (1.0, 0.3411764705882353, 0.3411764705882353),
        ),
        "alpha": ((0.0, 1, 1), (0.5, 1, 1), (1.0, 1, 1)),
    }  # #1E88E5 -> #ff0052
    red_blue_solid = LinearSegmentedColormap("RedBlue", cdict1)
    return red_blue_solid

