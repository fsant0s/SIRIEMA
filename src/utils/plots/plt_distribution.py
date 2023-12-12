import plotly.graph_objects as go
import plotly.io as pio

from config.definitions import ROOT_DIR

def distribution(df, path):
    # get the unique clusters
    clusters = df['cluster'].unique()
    colors = ['rgb(0, 0, 0)', 'rgb(100, 100, 100)', 
              'rgb(150, 150, 150)', 'rgb(200, 200, 200)',
              'rgb(255, 255, 255)']  # modified the last RGB value
    patterns = ['x', '/', '\\', '|', '-'] 

    fig = go.Figure()

    for i, cluster in enumerate(clusters):
        cluster_data = df[df['cluster'] == cluster]['elite_count']
        fig.add_trace(go.Histogram(x=cluster_data, 
                                   nbinsx=20, 
                                   name=f'Cluster {cluster}',
                                   histnorm='probability',  # normalize here
                                   marker_color=colors[i],
                                   opacity=1,
                                   marker_pattern_shape=patterns[i],  
                                  )
                     )
        
    fig.update_layout(
          # ... [other layout settings]
        legend=dict(
            # ... [other legend settings]
            #itemsizing='constant',  # Make the legend symbols of constant size
            itemwidth=40  # Adjust this value to change the legend symbol width
        )
	)
    

    tam_font = 60
    fig.update_layout(
        barmode='overlay',
        plot_bgcolor='rgba(250,250,250,0)',
        title_x=0.5,
        bargap=0.1,
        yaxis=dict(
            title='Density',
            titlefont_size=tam_font,
            tickfont_size=tam_font,
            range=[0,1],
            color='black',
            tickfont_family="Arial"
        ),
        xaxis=dict(
            title='Elite count',
            titlefont_size=tam_font,
            tickfont_size=tam_font,
            tickmode = 'array',
            color='black',
            tickfont_family="Arial"                  
        ),
        legend=dict(
            orientation="h",
            y=1.30,
            x=0,
            bgcolor='rgba(255, 255, 255, 1)',
            bordercolor='rgba(255, 255, 255, 1)',
            font=dict(
                family='Arial',
                size=tam_font + 6,
                color='black'
            )
        )
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black', mirror=True)

    pio.write_image(fig, ROOT_DIR + f"/src/results/{path[0]}/density/{path[1]}.png", height=1200, width=1100)
    fig.show()

# Example use-case (assuming df and path are correctly defined)
# distribution(df, path)
