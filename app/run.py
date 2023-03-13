"""This module implements a Flask web app
in which the the library/package disaster_response
is employed to (1) plot some disaster message
statistics and (2) predict the target categories of
a message input by the user.

Author: Mikel Sagardia
Date: 2023-03-13
"""
import json
import plotly
import pandas as pd
import numpy as np

from flask import Flask
from flask import render_template, request #, jsonify
from plotly.graph_objs import Bar, Layout #, Figure
from sqlalchemy import create_engine

from disaster_response import (load_validate_config,
                               load_validate_model,
                               DATABASE_TABLE_NAME)

# Falsk web app instance
app = Flask(__name__)

# Load config dictionary
config = load_validate_config(config_filename="config.yaml")

# Load data
engine = create_engine(f"sqlite:///{config['database_filepath']}")
df = pd.read_sql_table(DATABASE_TABLE_NAME, engine)

# Load model
model = load_validate_model(model_artifact=config['model_filepath'])

@app.route('/')
@app.route('/index')
def index():
    """Index page.
    This page first displays some plots related
    to the training dataset. Then, if the user inputs
    a message, it predicts the categories/targets."""
    graphs = []
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # 1. Distribution of Message Genres
    graph_one = []
    graph_one.append(
        Bar(x=genre_names,
            y=genre_counts)
        )
    layout_one = Layout(title = 'Distribution of Message Genres',
                        yaxis = {'title': "Count"},
                        xaxis = {'title': "Genre"})

    graphs.append(dict(data=graph_one, layout=layout_one))

    # 2. Distribution of Message Categories
    #category_name = list(df.columns[4:])
    category_names = config["target_columns"]
    category_counts = [np.sum(df[col]) for col in category_names]
    graph_two = []
    graph_two.append(Bar(x=category_names,
                         y=category_counts))
    layout_two =  Layout(title = 'Distribution of Message Categories',
                         yaxis = {'title': "Count"},
                         xaxis = {'title': "Genre"})

    graphs.append(dict(data=graph_two, layout=layout_two))

    # 3. Top 10 Message Categories
    #categories = df.iloc[:,4:]
    categories = df[category_names]
    category_means = categories.mean().sort_values(ascending=False)[1:11]

    graph_three = []
    graph_three.append(
            Bar(x=list(category_means.index), # category_names
                y=category_means*100))
    layout_three =  Layout(title = 'Top 10 Message Categories',
                yaxis = {
                    'title': "Percentage", 
                    'titlefont': {'color': 'black', 'size': 12}
                },
                xaxis = {
                'title': "Category", 
                'titlefont': {'color': 'black', 'size': 12},
                'tickangle':45,
                'automargin': True
                  }
                )
    
    graphs.append(dict(data=graph_three, layout=layout_three))

    # Encode Plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render web page with Plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    """Message classification page.
    This pages handles the user query and displays
    the model results."""
    # Save user input in query
    query = request.args.get('query', '') 

    # Use model to predict classification for query
    # FIXME: I need to pass a list with at least 2 elements (1 empty)
    # due to the ML pipeline definition
    classification_labels = model.predict(pd.DataFrame(data=[query, ""], columns=['message']))[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
