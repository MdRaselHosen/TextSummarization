



chat = "/Users/raselhosen/Desktop/textSummarization/chat.txt"

def extractChat(chat):
    userMessage = []
    AIMessage = []

    with open(chat,'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("User:"):
                userMessage.append(line[6:].strip())
            elif line.startswith("AI:"):
                AIMessage.append(line[3:].strip())

    return userMessage, AIMessage
extractedMessages = extractChat(chat)
print(extractedMessages)