import os
import copy
import dash
import numpy as np
from io import BytesIO
from flask import Flask
from dash import dcc
import base64
import dash_bootstrap_components as dbc
from dash import html
import plotly.graph_objects as go
import matplotlib.cm as cm
import scipy
import matplotlib.pyplot as plt

external_stylesheets = [dbc.themes.BOOTSTRAP,]

app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

app.title = 'TRAIL XAI Tool'

dummy_text = 'some stupid text'
dummy_img_url = 'https://lh3.googleusercontent.com/p/AF1QipPNfnW-ALDJ8LhK9QPjFhnhp6zr360oFvIrYBbH=w1080-h608-p-k-no-v0'
dummy_txt_tokens = ['[CLS]', 'some', 'stupid', 'text', '[SEP]']
dummy_img_tokens = ['[SEP]'] + [str(i) for i in range(49)]
dummy_img = np.random.random((100, 200, 3))

n_patches = 7

dummy_csm = np.random.random((len(dummy_text), n_patches))
dummy_txt_attn = np.random.random(len(dummy_txt_tokens))
dummy_img_attn = np.random.random((7, 7))

ATTN_MAP_SCALE_FACTOR = 64

heatmap_layout = dict(
    autosize=True,
    font=dict(color='black'),
    titlefont=dict(color='black', size=14),
    legend=dict(font=dict(size=8), orientation='h'),
)

img_overlay_layout = dict(
    barmode='stack',
    bargap=0,
    hovermode='closest',
    showlegend=False,
    autosize=False,
    margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(visible=False, fixedrange=True),
    yaxis=dict(visible=False, scaleanchor='x')  # constant aspect ratio
)


def show_csm(csm, title, txt_tokens, img_tokens):
    data = dict(
        type='heatmap',
        z=csm,
        colorbar=dict(thickness=10),
    )
    layout = heatmap_layout
    layout.update({
        'title': title,
        'title_font_size': 16,
        'title_x': 0.5, 'title_y': 0.99,
        'margin': {'t': 70, 'b': 10, 'l': 0, 'r': 0}
    })
    txt_attn_layout = {
        'xaxis': dict(
            tickmode='array',
            tickvals=list(range(len(txt_tokens))),
            ticktext=txt_tokens,
            tickangle=45,
            tickfont=dict(size=12),
            range=(-0.5, len(txt_tokens)-0.5),
            fixedrange=True),
        'yaxis': dict(
            tickmode='array',
            tickvals=list(range(len(txt_tokens))),
            ticktext=txt_tokens,
            tickfont=dict(size=12),
            range=(-0.5, len(txt_tokens)-0.5),
            fixedrange=True),
        'width': 600,
        'height': 600
    }
    img_attn_layout = {
        'xaxis': dict(range=(-0.5, len(img_tokens)-len(txt_tokens)-0.5)),
        'yaxis': dict(range=(-0.5, len(img_tokens)-len(txt_tokens)-0.5), scaleanchor='x'),
        'width': 900,
        'height': 900,
        'plot_bgcolor': 'rgba(0,0,0,0)',
    }
    figure = go.Figure(dict(data=data, layout=layout))

    return figure


def get_sep_indices(tokens):
    return [-1] + [i for i, x in enumerate(tokens) if x == '[SEP]']

def show_texts(tokens, colors='white'):
    seps = get_sep_indices(tokens)
    sentences = [tokens[i+1:j+1] for i, j in zip(seps[:-1], seps[1:])]
    print(seps, sentences)

    fig = go.Figure()
    annotations = []
    for sen_i, sentence in enumerate(sentences):
        word_lengths = list(map(len, sentence))
        fig.add_trace(go.Bar(
            x=word_lengths,  # TODO center at 0
            y=[sen_i] * len(sentence),
            orientation='h',
            marker_color=(colors if type(colors) is str else colors[sen_i]),
            marker_line=dict(color='rgba(255, 255, 255, 0)', width=0),
            hoverinfo='none'
        ))
        word_pos = np.cumsum(word_lengths) - np.array(word_lengths) / 2
        for word_i in range(len(sentence)):
            annotations.append(dict(
                xref='x', yref='y',
                x=word_pos[word_i], y=sen_i,
                text=sentence[word_i],
                showarrow=False
            ))
    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True, range=(len(sentences)+.3, -len(sentences)+.7))
    fig.update_layout(
        annotations=annotations,
        barmode='stack',
        autosize=False,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        showlegend=False,
        plot_bgcolor='white'
    )
    return fig

def plot_attn_from_txt(txt_tokens, txt_attn):

    # get colors by sentence
    colors = cm.get_cmap('Reds')(txt_attn * 120, 0.5)
    colors = ['rgba(' + ','.join(map(lambda x: str(int(x*255)), rgba[:3])) + ',' + str(rgba[3]) + ')'
               for rgba in colors]
    seps = get_sep_indices(txt_tokens)
    colors = [colors[i+1:j+1] for i, j in zip(seps[:-1], seps[1:])]
    txt_fig = show_texts(txt_tokens, colors)
    return txt_fig

def fig_to_uri(fig, close_all=True, **save_args):
    out_img = BytesIO()
    fig.savefig(out_img, format='jpeg', **save_args)
    if close_all:
        fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode('ascii').replace('\n', '')
    return 'data:image/jpeg;base64,{}'.format(encoded)

def show_img(img, opacity, bg, hw=None):
    img_height, img_width = hw if hw else (img.shape[0], img.shape[1])
    mfig, ax = plt.subplots(figsize=(img_width/100., img_height/100.), dpi=100)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    plt.axis('off')
    if hw:
        ax.imshow(img, cmap='jet', interpolation='nearest', aspect='auto')
    else:
        ax.imshow(img)
    img_uri = fig_to_uri(mfig)
    fig_width, fig_height = mfig.get_size_inches() * mfig.dpi

    fig = go.Figure()
    fig.update_xaxes(range=(0, fig_width))
    fig.update_yaxes(range=(0, fig_height))
    fig.update_layout(img_overlay_layout)
    fig.update_layout(
        autosize=True,
        plot_bgcolor=bg,
        paper_bgcolor=bg
    )
    fig.layout.images = []  # remove previous image
    fig.add_layout_image(dict(
        x=0, y=fig_height,
        sizex=fig_width, sizey=fig_height,
        xref='x', yref='y',
        opacity=opacity,
        sizing='stretch',
        source=img_uri
    ))
    return fig

def plot_image(image):
    return show_img(image, bg='black', opacity=1.0)

# just image
def display_ex_image(image):
    image_fig = plot_image(image)
    sm_image_fig = copy.deepcopy(image_fig)
    sm_image_fig.update_layout(height=256)
    return sm_image_fig#, sm_image_fig

# just overlay
def plot_attn_from_img(img, img_attn):
    img_attn = scipy.ndimage.zoom(img_attn, ATTN_MAP_SCALE_FACTOR, order=1)
    return show_img(img_attn, opacity=0.3, bg='rgba(0,0,0,0)', hw=img.shape[:2])


app.layout = html.Div(
    [   
        # text input

        dbc.Input(
            id='txt-input', type='text', placeholder="Input Text", value=dummy_text,
            debounce=True, style={'width': '50%'}),
        # img input (url)
        dbc.Input(id='img-input', type='url', placeholder="Input Image", value=dummy_img_url,
            debounce=True, style={'width': '50%'}),
        # cos similarity map
        dcc.Graph(id=f'csm-figure', figure=show_csm(dummy_csm, 'CSM', dummy_txt_tokens, dummy_img_tokens), ),
        
        # text with attention
        html.Div(
            [
                dcc.Graph(id=f'txt-attn-figure', figure=plot_attn_from_txt(dummy_txt_tokens, dummy_txt_attn), config={"displayModeBar": True}
                ),
            ], style={"width": "40%", "position": "absolute", "height": "450px"} 
        ),

        # # image with attention
        # html.Div(id="attn2img", children=[  # image
        #     html.Div(
        #         dcc.Graph(id="img-attn", config={"displayModeBar": False,}, figure=display_ex_image(dummy_img), style={"height": "450px"}),
        #         style={"width": "40%", "position": "absolute"}),
        #     html.Div(
        #         [dcc.Graph(id="img-attn-overlay", figure=plot_attn_from_img(dummy_img, dummy_img_attn), config={"displayModeBar": False})],
        #         style={"width": "40%", "position": "absolute", "height": "450px"}),
        # ]),


        html.Div(  # image display area
            [
                dcc.Graph(id="ex_sm_image", config={"displayModeBar": False}, figure=display_ex_image(dummy_img),
                        style={"width": "37%", "height": "256px", "margin-left": "4%", "position": "absolute"}),
                dcc.Graph(id="ex_img_attn_overlay", config={"displayModeBar": False},figure=plot_attn_from_img(dummy_img, dummy_img_attn),
                        style={"width": "37%", "height": "256px", "margin-left": "4%", "position": "absolute"})
            ], style={"margin-left": "50%"}
        ),
    ]
)


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=6006, debug=True, dev_tools_hot_reload=True, use_reloader=True)
