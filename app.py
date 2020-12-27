from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
import ast
from scripts.scrapper import get_shemantic_paper_html, extract_data
app = Flask(__name__)
api = Api(app)


class Parser(Resource):
    def get(self):
        parser = reqparse.RequestParser()  # initialize

        parser.add_argument('paper_id', required=True)  # add args

        args = parser.parse_args()
        paper_where = args['paper_id']
        soup = get_shemantic_paper_html(paper_where)
        data = extract_data(soup)
        # print(paper_where)

        return data, 200  # return data and 200 OK code


api.add_resource(Parser, '/parse')


if __name__ == '__main__':
    app.run()  # run our Flask app
