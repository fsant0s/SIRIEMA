import matplotlib.pyplot as plt
import seaborn as sns
from config.definitions import ROOT_DIR

def scatterplot(df, path):
    palette = {
        0: '#000000',  # Black (0% brightness)
        1: '#333333',  # Dark gray (20% brightness)
        2: '#666666',  # Gray (40% brightness)
        3: '#999999',  # Light gray (60% brightness)
        4: '#CCCCCC',  # Lighter gray (80% brightness)
    }
    
    palette = {
        0: '#000000',  # Black (0% brightness)
        1: '#999999',  # Dark gray (20% brightness)
        2: '#666666',  # Gray (40% brightness)
        3: '#999999',  # Light gray (60% brightness)
        4: '#CCCCCC',  # Lighter gray (80% brightness)
    }

	
    #palette = {
	#	0: 'red',     # Color for cluster 0
	#	1: 'blue',    # Color for cluster 1
	#	2: 'green',   # Color for cluster 2
	#	3: 'purple',  # Color for cluster 3
	#	4: 'black',   # Color for cluster 4
	#}

    plt.figure(figsize=(40, 40))
    plt.xticks([])  # Remove x-axis tick labels
    plt.yticks([])  # Remove y-axis tick labels
    plt.grid(True)

    sns.scatterplot(data=df, x=0, y=1, hue='cluster', palette=palette, s=1200) 
    tam_font = 170
    plt.xlabel('X', fontsize=tam_font)
    plt.ylabel('Y', fontsize=tam_font)
    
    # Create custom labels
    labels = [""]
    unique_clusters = df['cluster'].unique()
    for cluster in sorted(unique_clusters):
        labels.append(f'Cluster {cluster}')
    
    # Create custom handles with colors from the palette
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markersize=0)]
    for cluster in sorted(unique_clusters):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[cluster], markersize=100))
    
    # Creating the legend
    legend = plt.legend(prop={'size': tam_font}, handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(0, 1.15), frameon=True, ncol=3, handletextpad=0, columnspacing=0)
    for handle in legend.legendHandles:
        handle._sizes = [1000]

    grid_size = 15
    for spine in plt.gca().spines.values():
        spine.set_linewidth(grid_size)  # Adjust the line thickness as desired
    plt.grid(linewidth=grid_size)  # Adjust the linewidth as desired
	
#    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1, 
            hspace = 0, wspace = 0)
    #plt.margins(0.1,0.1)


    plt.tick_params(axis='both', labelsize=tam_font)
    plt.savefig(ROOT_DIR + f"/src/results/{path[0]}//scatterplot/{path[1]}.png", bbox_inches='tight', pad_inches=0)
    plt.show()

	#legend = plt.legend(prop={'size': tam_font}, loc='upper left', bbox_to_anchor=(0, 1.3), frameon=True, title="Clusters", ncol=5)




'''
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from config.definitions import ROOT_DIR
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd

def scatterplot(df, path):
	# Create a scatter plot with Plotly
	color_map = {
		0: 'rgb(255, 0, 0)',    # Red for cluster 0
		1: 'rgb(0, 255, 0)',    # Green for cluster 1
		2: 'rgb(0, 0, 255)',    # Blue for cluster 2
		3: 'rgb(255, 255, 0)',  # Yellow for cluster 3
		# Add more colors if you have more clusters
	}
	fig = px.scatter(df, x=0, y=1, symbol='cluster', color_discrete_map=color_map)


	tam_font = 60
	fig.update_layout(
		barmode='overlay',
		plot_bgcolor='rgba(3,250,250,0)',
		title_x=0.5,
		bargap=0.1,
		yaxis=dict(
			title='Y',
			titlefont_size=tam_font,
			tickfont_size=tam_font,
			#range=[0,1],
			color='black',
			tickfont_family="Arial"
		),
		xaxis=dict(
			title='X',
			titlefont_size=tam_font,
			tickfont_size=tam_font,
			tickmode = 'array',
			color='black',
			tickfont_family="Arial"                  
		),
		legend=dict(
			orientation="h",
			y=1.30,
			x = 0,
			bgcolor='rgba(255, 255, 255, 1)',
		 	bordercolor='rgba(255, 255, 255, 1)',
			font=dict(
				family='Arial',
				size= tam_font + 6,
				color='black'
			)
		)
	)

	fig.update_traces(marker=dict(color='rgb(0, 0, 0, 0)', opacity=0.7,  size=5))
	fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black', mirror=True)
	fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black', mirror=True)
	
	pio.write_image(fig, ROOT_DIR + f"/src/results/stability/{path[0]}/scatterplot/{path[1]}.png", height=1200, width=1100)
	fig.show()
'''