from flask import render_template, redirect, url_for, request
from markdown import markdown
from loguru import logger
from hordeling.flask import APP
import hordeling.exceptions as e

@logger.catch(reraise=True)
@APP.route('/')
# @cache.cached(timeout=300)
def index():
    with open(f'hordeling/templates/index.md') as index_file:
        index = index_file.read()
    findex = index.format()

    style = """<style>
        body {
            max-width: 120ex;
            margin: 0 auto;
            color: #333333;
            line-height: 1.4;
            font-family: sans-serif;
            padding: 1em;
        }
        </style>
    """

    head = f"""<head>
    <title>AI Hordeling</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    {style}
    </head>
    """
    return(head + markdown(findex))
