from flask import Flask, render_template, request
import json

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    print('data', data)
    messages = data['messages']
    processed_messages = [message.upper() for message in messages]
    response_data = {
        'processed_messages': processed_messages,
    }
    print(response_data)
    return json.dumps(response_data)


if __name__ == '__main__':
   app.run(debug=True)