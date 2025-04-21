

import json


def find_diff_emojies(file_path):
    """
    Find the difference between two emoji files.
    
    Args:
        file_path (str): Path to the emoji file.
        
    Returns:
        list: List of emojis that are in the first file but not in the second.
    """
    messages = []
    with open(file_path, 'r') as f:
        messages = json.load(f)
        reaction_set = set()
        for message in messages:
            if 'reactions' in message:
                for reaction in message['reactions']:
                    reaction_set.add((
                        reaction['type'] if 'type' in reaction else None,
                        reaction['pack_id'] if 'pack_id' in reaction else None,
                        reaction['pack_index'] if 'pack_index' in reaction else None,
                        reaction['code'] if 'code' in reaction else None,
                    ))
                    if 'type' in reaction and reaction['type'] == 'unicode':
                        print(f"EMOIE INFO \n {reaction}")
                        print(f"EMJOIE CODE \n {str(reaction['code'])}")
                        print(message['text'])
                        print("\n\n")


        return reaction_set
    
if __name__ == "__main__":
    # print("running")
    # # Load your messages from a JSON file
    # reaction_set = find_diff_emojies('THIS_YEAR_MESSAGES.json')
    # for info in reaction_set:
    #     if info[0] == 'unicode':
    #         print(f"EMOIE INFO \n {info}")
    #         print(f"EMJOIE CODE \n {str(info[3])}")

    for i in range(2):
        print("\u2753")
        print("\ud83d\udc4e".encode('utf-16', 'surrogatepass').decode('utf-16'))

    # Assuming the second file is 'emojis2.txt'
   


    
    
