import os
import argparse
from flask import Flask, request, render_template
from waitress import serve
from request_classifier.download_models import download_models

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("command", type=str, 
                        help="use 'start' to run the app")
arg_parser.add_argument('-p', '--port', default=5003,
                        help="port for the app to use")
args = arg_parser.parse_args()

if args.command == 'start':
    from request_classifier.classify_request_v2 import *

    app = Flask(__name__)
    FILE_DIR = os.path.abspath(os.path.dirname(__file__))

    @app.route('/')
    def query_form():
        return render_template('index.html', checked='on')

    @app.route('/', methods=['GET', 'POST'])
    def run_classifier():
        multi_label_flag = 'off'

        if request.method == 'GET':
            query_text = request.args.get('query', default='no query specified', type=str)

        else:
            query_text = request.form['query']
            multi_label_flag = request.form['multi_label']
            # print (multi_label_flag)
        if multi_label_flag == 'off':
            result = run_prediction_single_label([query_text, ''])[0]
        else:
            result = run_prediction([query_text, ''])[0]
        # result = run_prediction([query_text, ''])[0]
        print (result)
        # return result
        return render_template('index.html', text_query=query_text, out_class=result, checked=multi_label_flag)


    @app.route('/get_category', methods=['POST'])
    def ajax_response():
        multi_label_flag = 'false'
        print ('lol')
        if request.method == 'GET':
            query_text = request.args.get('query', default='no query specified', type=str)

        else:
            print (request.form)
            query_text = request.form['query']
            multi_label_flag = request.form['multi_label']
            print (multi_label_flag)
        if multi_label_flag == 'false':
            result = run_prediction_single_label([query_text, ''])[0]
        else:
            result = run_prediction([query_text, ''])[0]
        # return result
        # print (type(query_text))
        # result = run_prediction([query_text, ''])[0]
        # out = [class_dict[r] for r in result]
        return str(result)

def main():
    if args.command == 'download_models':
        download_models()
    if args.command == 'start':
        # serve(app, host="0.0.0.0", port=int(os.environ.get('PORT', 5001)))
        #app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5003)))
        serve(app, host="0.0.0.0", port=int(os.environ.get('PORT', args.port)))

if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5003)))
    serve(app, host="0.0.0.0", port=int(os.environ.get('PORT', args.port)))
    # print ('app started')
    # while True:
    #     query = input()
    #     print (classify_request(query))

