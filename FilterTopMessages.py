import json

def get_top_messages(file_path, top_n = 30):
    """ Gets top n messages, longest messages"""
    print("Top messages: \n\n\n")

    with open(file_path) as f:
        data = json.load(f)
        messages = data
        sorted_messages = sorted(messages, key=lambda x: sum(len(react['user_ids']) for react in x['reactions']) if 'reactions' in x else 0, reverse=True)[:top_n]
        for message in sorted_messages:
            print(message['attachments'])
            print(message['text'])
            print(sum(len(react['user_ids']) for react in message['reactions']))
            print('------')

if __name__ == '__main__':
    file_path = 'training.json'
    get_top_messages(file_path)