import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from src.model import predict_image, compute_entropy
from src.etl import load_image, get_image_files
from src.utils import image_to_base64
import os

# Crear la aplicación Dash
app = Dash(__name__, title="ImageNet Analyzer")

# Obtener la lista de imágenes en la carpeta assets
image_folder = 'assets/images/'
image_files = get_image_files(image_folder)
image_file_ids = [img.replace('.', '_') for img in image_files]

# Layout de la aplicación
app.layout = html.Div(
    style={"text-align": "center", "padding": "50px", "background-color": "#222", "color": "#fff"},
    children=[
        html.H1("ImageNet Analyzer"),
        
        html.Div(
            children=[
                html.H3("Selecciona una Imagen para Procesar"),
                html.Div(
                    id="image-gallery",
                    children=[
                        html.Div(
                            children=html.Img(
                                src=f"assets/images/{img}",
                                id=f"image-{img.replace('.', '_')}",
                                style={"width": "100px", "height": "100px", "margin": "10px", "cursor": "pointer", "border": "2px solid #555"}
                            ),
                            style={"display": "inline-block"}
                        ) for img in image_files
                    ],
                    style={"display": "flex", "flex-wrap": "wrap", "justify-content": "center"}
                ),
            ],
            style={"margin-bottom": "50px"}
        ),

        html.Div(
            children=[
                html.H3("Imagen Procesada por el Modelo"),
                html.Img(id="processed-image", src="", style={"width": "40%", "border": "2px solid #007bff", "border-radius": "8px"})
            ]
        ),

        html.Div(
            children=[
                html.H4("Top 3 Clases Predichas:"),
                html.Div(id="predicted-classes", style={"font-weight": "bold", "font-size": "20px", "color": "#007bff"})
            ],
            style={"margin-top": "20px"}
        ),

        html.Div(children=[dcc.Graph(id="probability-graph")], style={"margin-top": "20px"}),

        html.Div(
            children=[
                html.H4("Valor de la Entropía:"),
                html.Div(id="entropy-value", style={"font-weight": "bold", "font-size": "20px", "color": "#007bff"})
            ],
            style={"margin-top": "20px"}
        ),
    ]
)

@app.callback(
    [Output("processed-image", "src"),
     Output("predicted-classes", "children"),
     Output("probability-graph", "figure"),
     Output("entropy-value", "children")],
    [Input(f"image-{img.replace('.', '_')}", "n_clicks") for img in image_files]
)
def update_image(*args):
    triggered_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if not triggered_id:
        return "", "", {}, ""

    selected_image = triggered_id.split('-')[1].replace('_', '.')
    img_path = f"{image_folder}{selected_image}"
    img = load_image(img_path)

    top3_class_names, top3_probs, probabilities = predict_image(img)
    img_base64 = image_to_base64(img_path)
    
    predicted_classes = "".join([f"{name}: {prob*100:.2f}%\t" for name, prob in zip(top3_class_names, top3_probs)])

    figure = {
        'data': [go.Bar(x=top3_class_names, y=top3_probs * 100, marker={'color': 'rgb(55, 83, 109)'})],
        'layout': go.Layout(title="Top 3 Predicciones", xaxis={'title': 'Clase'}, yaxis={'title': 'Probabilidad (%)'}, showlegend=False)
    }

    entropy_value = f"{compute_entropy(probabilities):.4f}"

    return f"data:image/jpeg;base64,{img_base64}", predicted_classes, figure, entropy_value

if __name__ == "__main__":
    app.run_server(debug=True)