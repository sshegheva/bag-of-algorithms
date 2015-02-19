import matplotlib.pyplot as plt


def draw_monalisa(monalisa_df, solution):
    xpairs = []
    ypairs = []

    for i in range(len(solution) - 1):
        xpairs.append([monalisa_df.iloc[solution[i]]['X'], monalisa_df.iloc[solution[i+1]]['X']])
        ypairs.append([monalisa_df.iloc[solution[i]]['Y'], monalisa_df.iloc[solution[i+1]]['Y']])

    for xends,yends in zip(xpairs,ypairs):
        plt.plot(xends, yends ,'b-', alpha=0.1)