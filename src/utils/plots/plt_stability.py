import plotly.graph_objects as go
import plotly.io as pio

import numpy as np
import pandas as pd

from config.definitions import ROOT_DIR

def plt_bar_stability_samples(data, path, title):
	with_title = False
	fig_title = title if with_title else ""
	clusters = list(pd.unique(data['sample']))
	
	mean_struc = data.loc[data['type'] == 'Structured']['mean'].reset_index(drop=True)
	ci_upper_struc = data.loc[data['type'] == 'Structured']['ci_upper'].reset_index(drop=True)
	ci_lower_struc = data.loc[data['type'] == 'Structured']['ci_lower'].reset_index(drop=True)
	
	mean_unstruc = data.loc[data['type'] == 'Unstructured']['mean'].reset_index(drop=True)
	ci_upper_unstruc = data.loc[data['type'] == 'Unstructured']['ci_upper'].reset_index(drop=True)
	ci_lower_unstruc = data.loc[data['type'] == 'Unstructured']['ci_lower'].reset_index(drop=True)

	mean_merged = data.loc[data['type'] == 'Structured & Unstructured']['mean'].reset_index(drop=True)
	ci_upper_merged = data.loc[data['type'] == 'Structured & Unstructured']['ci_upper'].reset_index(drop=True)
	ci_lower_merged = data.loc[data['type'] == 'Structured & Unstructured']['ci_lower'].reset_index(drop=True)
        
	mean_DECF = data.loc[data['type'] == 'DECF']['mean'].reset_index(drop=True)
	ci_upper_DECF = data.loc[data['type'] == 'DECF']['ci_upper'].reset_index(drop=True)
	ci_lower_DECF = data.loc[data['type'] == 'DECF']['ci_lower'].reset_index(drop=True)

	mean_multimodal = data.loc[data['type'] == 'Gating']['mean'].reset_index(drop=True)
	ci_upper_multimodal = data.loc[data['type'] == 'Gating']['ci_upper'].reset_index(drop=True)
	ci_lower_multimodal = data.loc[data['type'] == 'Gating']['ci_lower'].reset_index(drop=True)

	fig = go.Figure()
	fig.add_trace(
		go.Bar(
			x=list(range(0, len(clusters)*5,5)),
			y=mean_merged,
			name='CD - SU',
			marker_color='rgb(1000, 1000, 1000)',

			marker_line=dict(width=2, color='black'),
			marker_pattern_shape="+",
			error_y=dict(
				type='data',
				thickness = 2,
				symmetric=False,
				array=np.subtract(ci_upper_merged, mean_merged),
				arrayminus=np.subtract(mean_merged, ci_lower_merged)
			)
		)
	)
        
	fig.add_trace(
		go.Bar(
			x=list(range(0, len(clusters)*5,5)),
			y=mean_struc,
			name='Structured',
			marker_color='rgb(200, 200, 200)',
			marker_pattern_shape="/",
			marker_line=dict(width=2, color='black'),
			error_y=dict(
				thickness = 2,
				type='data',
				symmetric=False,
				array=np.subtract(ci_upper_struc, mean_struc),
				arrayminus=np.subtract(mean_struc, ci_lower_struc)
			)
		)
	)

	fig.add_trace(
		go.Bar(
			x=list(range(0, len(clusters)*5,5)),
			y=mean_unstruc,
			name='Unstructured',
			marker_color='rgb(100, 100, 100)',
			marker_line=dict(width=2, color='black'),
			marker_pattern_shape="x",
			error_y=dict(
				thickness = 2,
				type='data',
				symmetric=False,
				array=np.subtract(ci_upper_unstruc, mean_unstruc),
				arrayminus=np.subtract(mean_unstruc, ci_lower_unstruc)
			)	
		)
	)
        
	fig.add_trace(
		go.Bar(
			x=list(range(0, len(clusters)*5,5)),
			y=mean_DECF,
			name='Mixed DEC + SU',
			marker_color='rgb(50, 50, 50)',
			marker_line=dict(width=2, color='black'),
			marker_pattern_shape="\\",
			error_y=dict(
				type='data',
				thickness = 2,
				symmetric=False,
				array=np.subtract(ci_upper_DECF, mean_DECF),
				arrayminus=np.subtract(mean_DECF, ci_lower_DECF)
			)
		)
	)

	fig.add_trace(
		go.Bar(
			x=list(range(0, len(clusters)*5,5)),
			y=mean_multimodal,
			name='Our proposal',
			marker_color='rgb(0, 0, 0)',
			marker_line=dict(width=2, color='black'),
			marker_pattern_shape=".",
			error_y=dict(
				type='data',
				thickness = 2,
				symmetric=False,
				array=np.subtract(ci_upper_multimodal, mean_multimodal),
				arrayminus=np.subtract(mean_multimodal, ci_lower_multimodal)
			)
		)
	)
        
	fig.update_layout(
		title= "<b>" + fig_title + "</b>",
		title_font=dict(
			family='Arial, sans-serif',
			size=24,
			color='Black'  # Set the title color here
   		),
		title_x=0.5,
		#paper_bgcolor='rgb(250,250,250)',
   	 	plot_bgcolor='rgba(250,250,250,0)',
		yaxis=dict(
			title='<b>Stability</b>',
			titlefont_size=16,
			tickfont_size=14,
			range=[0,1],
			color='black',
            tickfont_family="Arial Black"
		),
		xaxis=dict(
			title='<b>Samples</b>',
			titlefont_size=16,
			tickfont_size=14,
			tickmode = 'array',
			ticktext = clusters,
            color='black',
      		tickvals = list(range(0, len(clusters)*5,5)),
            tickfont_family="Arial Black"                  
		),
		legend=dict(
			orientation="h",
			y=1.15,
			bgcolor='rgba(255, 255, 255, 1)',
			bordercolor='rgba(255, 255, 255, 1)',
            font=dict(
				family='Arial, sans-serif',
				size=16,  # Adjust the size of the legend text here
				color='black'
			)
		),
		barmode='group',
		bargap=0.15, # gap between bars of adjacent location coordinates.
		bargroupgap=0.1 # gap between bars of the same location coordinate.
	)


	path_title = "with_title" if with_title else "without_title"
	fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black', mirror=True)
	fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black', mirror=True)
	#pio.write_image(fig, ROOT_DIR + f"/src/results/{path[0]}/stability/{path[1]}/{path_title}/{path[1]}-{title}.png", width=1200)
	fig.show()


def plt_bar_stability_structured(data, title):
	
	clusters = list(pd.unique(data.cluster))
	
	mean_adjusted_rand_score = data.loc[data['measure'] == 'adjusted_rand_score']['mean'].reset_index(drop=True)
	ci_upper_adjusted_rand_score = data.loc[data['measure'] == 'adjusted_rand_score']['ci_upper'].reset_index(drop=True)
	ci_lower_adjusted_rand_score = data.loc[data['measure'] == 'adjusted_rand_score']['ci_lower'].reset_index(drop=True)
	
	mean_adjusted_mutual_info_score = data.loc[data['measure'] == 'adjusted_mutual_info_score']['mean'].reset_index(drop=True)
	ci_upper_adjusted_mutual_info_score = data.loc[data['measure'] == 'adjusted_mutual_info_score']['ci_upper'].reset_index(drop=True)
	ci_lower_adjusted_mutual_info_score = data.loc[data['measure'] == 'adjusted_mutual_info_score']['ci_lower'].reset_index(drop=True)

	mean_bagclust = data.loc[data['measure'] == 'bagclust']['mean'].reset_index(drop=True)
	ci_upper_bagclust = data.loc[data['measure'] == 'bagclust']['ci_upper'].reset_index(drop=True)
	ci_lower_bagclust = data.loc[data['measure'] == 'bagclust']['ci_lower'].reset_index(drop=True)

	mean_han = data.loc[data['measure'] == 'han']['mean'].reset_index(drop=True)
	ci_upper_han = data.loc[data['measure'] == 'han']['ci_upper'].reset_index(drop=True)
	ci_lower_han = data.loc[data['measure'] == 'han']['ci_lower'].reset_index(drop=True)


	mean_OTclust = data.loc[data['measure'] == 'OTclust']['mean'].reset_index(drop=True)
	ci_upper_OTclust = data.loc[data['measure'] == 'OTclust']['ci_upper'].reset_index(drop=True)
	ci_lower_OTclust = data.loc[data['measure'] == 'OTclust']['ci_lower'].reset_index(drop=True)

	fig = go.Figure()
	fig.add_trace(
		go.Bar(
			x=list(range(0, len(clusters)*5,5)),
			y=mean_adjusted_rand_score,
			name='Adjusted mutual info score',
			marker_color='rgb(200, 200, 200)',
			marker_pattern_shape="/",
			marker_line=dict(width=0.5, color='black'),
			error_y=dict(
				thickness = 1.3,
				type='data',
				symmetric=False,
				array=np.subtract(ci_upper_adjusted_rand_score, mean_adjusted_rand_score),
				arrayminus=np.subtract(mean_adjusted_rand_score, ci_lower_adjusted_rand_score)
			)
		)
	)

	fig.add_trace(
		go.Bar(
			x=list(range(0, len(clusters)*5,5)),
			y=mean_adjusted_mutual_info_score,
			name='Adjusted Rand Score',
			marker_color='rgb(180, 180, 180)',
			marker_line=dict(width=0.5, color='black'),
			marker_pattern_shape="x",
			error_y=dict(
				thickness = 1.3,
				type='data',
				symmetric=False,
				array=np.subtract(ci_upper_adjusted_mutual_info_score, mean_adjusted_mutual_info_score),
				arrayminus=np.subtract(mean_adjusted_mutual_info_score, ci_lower_adjusted_mutual_info_score)
			)	
		)
	)

	fig.add_trace(
		go.Bar(
			x=list(range(0, len(clusters)*5,5)),
			y=mean_bagclust,
			name='Bag Clust',
			marker_color='rgb(160, 160, 160)',
			marker_line=dict(width=0.5, color='black'),
			marker_pattern_shape="+",
			error_y=dict(
				type='data',
				thickness = 1.3,
				symmetric=False,
				array=np.subtract(ci_upper_bagclust, mean_bagclust),
				arrayminus=np.subtract(mean_bagclust, ci_lower_bagclust)
			)
		)
	)

	fig.add_trace(
		go.Bar(
			x=list(range(0, len(clusters)*5,5)),
			y=mean_han,
			name='Han',
			marker_color='rgb(140, 140, 140)',
			marker_line=dict(width=0.5, color='black'),
			marker_pattern_shape="\\",
			error_y=dict(
				type='data',
				thickness = 1.3,
				symmetric=False,
				array=np.subtract(ci_upper_han, mean_han),
				arrayminus=np.subtract(mean_han, ci_lower_han)
			)
		)
	)

	fig.add_trace(
		go.Bar(
			x=list(range(0, len(clusters)*5,5)),
			y=mean_OTclust,
			name='OTclust',
			marker_color='rgb(120, 120, 120)',
			marker_line=dict(width=0.5, color='black'),
			marker_pattern_shape=".",
			error_y=dict(
				type='data',
				thickness = 1.3,
				symmetric=False,
				array=np.subtract(ci_upper_OTclust, mean_OTclust),
				arrayminus=np.subtract(mean_OTclust, ci_lower_OTclust)
			)
		)
	)

	fig.update_layout(
		title=title,
		xaxis_tickfont_size=14,
		#paper_bgcolor='rgb(250,250,250)',
    	plot_bgcolor='rgba(250,250,250,0)',
		yaxis=dict(
			title='Stability',
			titlefont_size=16,
			tickfont_size=14,
			range=[0,1]
		),
		xaxis=dict(
			title='Cluster',
			titlefont_size=16,
			tickfont_size=14,
			tickmode = 'array',
			ticktext = clusters,
      tickvals = list(range(0, len(clusters)*5,5)),
		),
		legend=dict(
			x=1.0,
			y=1.0,
			bgcolor='rgba(255, 255, 255, 0)',
			bordercolor='rgba(255, 255, 255, 0)'
		),
		barmode='group',
		bargap=0.15, # gap between bars of adjacent location coordinates.
		bargroupgap=0.1 # gap between bars of the same location coordinate.
	)

	# Change grid color and axis colors
	fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black', mirror=True)
	fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black', mirror=True)

	fig.show()


def plt_bar_stability_unstructured(data, title):

	clusters = pd.unique(data.cluster)

	mean_tfidf = data.loc[data['algorithm'] == 'tfidf']['mean'].reset_index(drop=True)
	ci_upper_tfidf = data.loc[data['algorithm'] == 'tfidf']['ci_upper'].reset_index(drop=True)
	ci_lower_tfidf = data.loc[data['algorithm'] == 'tfidf']['ci_lower'].reset_index(drop=True)
	
	mean_doc2word = data.loc[data['algorithm'] == 'doc2word']['mean'].reset_index(drop=True)
	ci_upper_doc2word = data.loc[data['algorithm'] == 'doc2word']['ci_upper'].reset_index(drop=True)
	ci_lower_doc2word = data.loc[data['algorithm'] == 'doc2word']['ci_lower'].reset_index(drop=True)

	mean_bert_imdb = data.loc[data['algorithm'] == 'bert_imdb']['mean'].reset_index(drop=True)
	ci_upper_bert_imdb = data.loc[data['algorithm'] == 'bert_imdb']['ci_upper'].reset_index(drop=True)
	ci_lower_bert_imdb = data.loc[data['algorithm'] == 'bert_imdb']['ci_lower'].reset_index(drop=True)

	mean_roberta = data.loc[data['algorithm'] == 'roberta_imdb']['mean'].reset_index(drop=True)
	ci_upper_roberta = data.loc[data['algorithm'] == 'roberta_imdb']['ci_upper'].reset_index(drop=True)
	ci_lower_roberta = data.loc[data['algorithm'] == 'roberta_imdb']['ci_lower'].reset_index(drop=True)


	fig = go.Figure()
	fig.add_trace(
		go.Bar(
			x=clusters,
			y=mean_tfidf,
			name='TF-IDF',
			marker_color='rgb(200, 200, 200)',
			marker_pattern_shape="/",
			marker_line=dict(width=0.5, color='black'),
			error_y=dict(
				thickness = 1.3,
				type='data',
				symmetric=False,
				array=np.subtract(ci_upper_tfidf, mean_tfidf),
				arrayminus=np.subtract(mean_tfidf, ci_lower_tfidf)
			)
		)
	)

	fig.add_trace(
		go.Bar(
			x=clusters,
			y=mean_doc2word,
			name='Doc2Word',
			marker_color='rgb(180, 180, 180)',
			marker_line=dict(width=0.5, color='black'),
			marker_pattern_shape="x",
			error_y=dict(
				thickness = 1.3,
				type='data',
				symmetric=False,
				array=np.subtract(ci_upper_doc2word, mean_doc2word),
				arrayminus=np.subtract(mean_doc2word, ci_lower_doc2word)
			)	
		)
	)

	fig.add_trace(
		go.Bar(
			x=clusters,
			y=mean_bert_imdb,
			name='BERT',
			marker_color='rgb(160, 160, 160)',
			marker_line=dict(width=0.5, color='black'),
			marker_pattern_shape="+",
			error_y=dict(
				type='data',
				thickness = 1.3,
				symmetric=False,
				array=np.subtract(ci_upper_bert_imdb, ci_upper_bert_imdb),
				arrayminus=np.subtract(ci_upper_bert_imdb, ci_lower_bert_imdb)
			)
		)
	)

	fig.add_trace(
		go.Bar(
			x=clusters,
			y=mean_roberta,
			name='roBERTa',
			marker_color='rgb(140, 140, 140)',
			marker_line=dict(width=0.5, color='black'),
			marker_pattern_shape="\\",
			error_y=dict(
				type='data',
				thickness = 1.3,
				symmetric=False,
				array=np.subtract(ci_upper_roberta, mean_roberta),
				arrayminus=np.subtract(mean_roberta, ci_lower_roberta)
			)
		)
	)

	fig.update_layout(
		title=title,
		xaxis_tickfont_size=14,
		#paper_bgcolor='rgb(250,250,250)',
    	plot_bgcolor='rgba(250,250,250,0)',
		yaxis=dict(
			title='Stability',
			titlefont_size=16,
			tickfont_size=14,
			range=[0,1]
		),
		xaxis=dict(
			title='Cluster',
			titlefont_size=16,
			tickfont_size=14,
		),
		legend=dict(
			x=1.0,
			y=1.0,
			bgcolor='rgba(255, 255, 255, 0)',
			bordercolor='rgba(255, 255, 255, 0)'
		),
		barmode='group',
		bargap=0.15, # gap between bars of adjacent location coordinates.
		bargroupgap=0.1 # gap between bars of the same location coordinate.
	)

	# Change grid color and axis colors
	fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black', mirror=True)
	fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black', mirror=True)

	fig.show()

def plot_line_stability(data, ALGORITHMS, CLUSTERS, method_name):
    pallete = [
        'rgb(200, 200, 200)',
        'rgb(180, 180, 180)',
        'rgb(160, 160, 160)',
        'rgb(140, 140, 140)',
        'rgb(120, 120, 120)'
    ]
    symbols = ['diamond', 'circle', 'square', 'cross', 'x']
    names = {
        'tfidf': 'TF-IDF',
        'doc2word': 'Doc2Word',
        'bert': 'BERT',
        'roberta': 'roBERTa'
    }
    fig = go.Figure()
    for i, alg in enumerate(ALGORITHMS):
        data_alg = data[data['algorithm'] == alg]
        fig.add_trace(  
            go.Scatter(
                name=names[alg],
                connectgaps=True,
                x= list(range(0, len(CLUSTERS)*5,5)),
                marker=dict(
                    color=pallete[i],
                    symbol=symbols[i],
                    size=8,
                    line=dict(
                        #color='MediumPurple',
                        width=1
                    )
                ),
                line_color=pallete[i],
                #marker_pattern_shape="x",
                y= data_alg['mean'],
                mode = 'lines+markers',
                line=dict(shape='linear'),
                customdata= np.stack((data_alg.ci_upper, data_alg.ci_lower), axis=-1),
                hovertemplate='<br>'.join([
                    'Cluster: %{x}',
                    'Stability: %{y:.2f}',
                    'CI upper: %{customdata[0]:.2f}',
                    'CI lower: %{customdata[1]:.2f}',
                ]),
            )
        )
        fig.add_trace(
            go.Scatter(
                name='95% CI Upper',
                x=list(range(0, len(CLUSTERS)*5,5)),
                y=data_alg.ci_upper,
                mode='lines',
                marker=dict(color='#444'),
                line=dict(width=0),
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                name='95% CI Lower',
                x=list(range(0, len(CLUSTERS)*5,5)),
                y=data_alg.ci_lower,
                marker=dict(color='#444'),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            )
        )

    fig.update_layout(
        title=method_name,
        legend_title_text='Algorithms',
        yaxis=dict(
			title='Stability',
			titlefont_size=16,
			tickfont_size=14,
			range=[0,1]
		),
        plot_bgcolor='rgba(250,250,250,0)',
        xaxis=dict(
            #range=[-2, CLUSTERS[-1]+1],
            tickmode = 'array',
            tickvals = list(range(0, len(CLUSTERS)*5,5)),
            ticktext = CLUSTERS,
            title='Cluster',
		    titlefont_size=16,
		    tickfont_size=14,
        ),
        hovermode='closest',
    )

    fig.update_yaxes(rangemode='tozero')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black', mirror=True)
    fig.show()

def plot_line_stability_by_samples(data, samples, method_name):
    symbols = ['diamond', 'circle', 'square', 'cross']
    type = ['Structured', 'Unstructured', 'Structured & Unstructured', 'Gating']
    pallete = [
        'rgb(200, 200, 200)',
        'rgb(180, 180, 180)',
        'rgb(160, 160, 160)',
        'rgb(140, 140, 140)',
        'rgb(120, 120, 120)'
    ]
    fig = go.Figure()
    for i, type in enumerate(type):
        data_alg = data[data['type'] == type]
        fig.add_trace(  
            go.Scatter(
                name=type,
                connectgaps=True,
                x = list(range(0, len(samples)*5,5)),
                marker=dict(
                    color=pallete[i],
                    symbol=symbols[i],
                    size=8,
                    line=dict(
                        #color='MediumPurple',
                        width=1
                    )
                ),
                line_color=pallete[i],
                #marker_pattern_shape="x",
                y= data_alg['mean'],
                mode = 'lines+markers',
                line=dict(shape='linear'),
                customdata= np.stack((data_alg.ci_upper, data_alg.ci_lower), axis=-1),
                hovertemplate='<br>'.join([
                    'Sample: %{x}',
                    'Stability: %{y:.2f}',
                    'CI upper: %{customdata[0]:.2f}',
                    'CI lower: %{customdata[1]:.2f}',
                ]),
            )
        )
        fig.add_trace(
            go.Scatter(
                name='95% CI Upper',
                x=list(range(0, len(samples)*5,5)),
                y=data_alg.ci_upper,
                mode='lines',
                marker=dict(color='#444'),
                line=dict(width=0),
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                name='95% CI Lower',
                x=list(range(0, len(samples)*5,5)),
                y=data_alg.ci_lower,
                marker=dict(color='#444'),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            )
        )

    fig.update_layout(
        title=method_name,
        legend_title_text='Data type',
        yaxis=dict(
			title='Stability',
			titlefont_size=16,
			tickfont_size=14,
			range=[0,1]
		),
        plot_bgcolor='rgba(250,250,250,0)',
        xaxis=dict(
            #range=[-2, CLUSTERS[-1]+1],
            tickmode = 'array',
            tickvals = list(range(0, len(samples)*5,5)),
            ticktext = samples,
            title='Number of samples',
		    titlefont_size=16,
		    tickfont_size=14,
        ),
        hovermode='closest',
    )

    fig.update_yaxes(rangemode='tozero')
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black', mirror=True)
    fig.show()


