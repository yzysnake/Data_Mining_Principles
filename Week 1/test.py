import requests

# Fetching the text from Project Gutenberg
url = "https://www.gutenberg.org/files/215/215-h/215-h.htm"
text = requests.get(url).text

# Counting the words in the text
word_count = len(text.split())

print(word_count)