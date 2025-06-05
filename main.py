from graphrag_agent import createAgent
import csv


def extract_conversations(csv_file_path):
    conversation_dict = {}
    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            conv_id = row.get("Conversation Id")
            message = row.get("Messages")
            if conv_id and message:
                conversation_dict[conv_id] = message
    return conversation_dict

def save_dict_to_csv(data_dict):
    with open('output.csv', mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the header
        writer.writerow(['Conversation Id', 'Policies'])
        
        # Write each key-value pair
        for key, value in data_dict.items():
            writer.writerow([key, value])


if __name__ == "__main__":
    csv_path = "example.csv"
    conversations = extract_conversations(csv_path)
    policies_dict = {}
    agent = createAgent()
    i = 0

    for key, value in conversations.items():
        policies = agent.invoke({"input": value})
        policies_dict[key] = policies["output"]
        i += 1
        if(i == 10):
            break

    save_dict_to_csv(policies_dict)
