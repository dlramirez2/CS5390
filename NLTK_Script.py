import nltk
##To open URLs and clean up HTML markup
from bs4 import BeautifulSoup
#Library to scrape info from websites
import requests
from nltk.text import Text

'''
clean_website() method used to strip URL text of HTML markup and Empty Lines.
Used BeautifulSoup Library to facilitate the operations.
'''
def clean_website(website_html):
    soup = BeautifulSoup(website_html,'html.parser')
    clean_html = soup.get_text(strip=True)
    return clean_html

'''
read_url() method used to :
1. Read website of IMDb User Reviews(in this case); define website's URL in the
 url object.
2. Use a Tokenizer to tokenize sentence
 " A token is the technical name for a sequence of characters" that is treated as a group
3. Translating the tokens into a Text Object. This object is used to explore texts using the
 nltk library methods.
 Example output:
    <Text: That '70s Show (TV Series 1998-2006 - That...>
'''
def read_url(url):

    #1
    html_text = requests.get(url).text
    #2
    t = nltk.tokenize.WhitespaceTokenizer()
    clean_html = clean_website(html_text)
    #3
    html_text = Text(t.tokenize(clean_html))
    return html_text
'''
If the html file of a website is available, we can also use this file to analyze the website contents.
This procedure of downloading website content has the disadvantage of not having update website content.
'''
def read_file(file_path):
    with open(file_path,'r') as f:
        contents = f.read()
        clean_html = clean_website(contents)
    t = nltk.tokenize.WhitespaceTokenizer()
    html_text = Text(t.tokenize(clean_html))
    return html_text

'''Define a url to analyze:'''
#html_text = read_url("https://www.imdb.com/title/tt0165598/reviews?ref_=tt_urv")

'''Define a file path to analyze as a read_file() method argument.'''
#html_text = read_file("That '70s Show.html")
html_text = read_file("JerseyShore.html")


'''
The concordance operation allows us to search for tokens in which the predefined word
is present within the Text object.

In this case for example, we are trying to search for instances where predefined "positive" words
are present in the reviews.
'''
#html_text.concordance("good")
#html_text.concordance("funny")

'''
In the next case, we are trying to search for instances of "negative" words in the reviews.
'''
#html_text.concordance("terrible")
#html_text.concordance("hate")
#html_text.concordance("bad")


'''
Frequency distribution NLTK operation. This operation can help us calculate the most common words and how many times a word
is used in our data.
'''
freq_dist = nltk.FreqDist(html_text)
#print(freq_dist.most_common(50))
print('Great: ' + str(freq_dist['great']))
print('Embarrassing: ' + str(freq_dist['embarrassing']))


'''
Resources:
    http://www.nltk.org/book/ch01.html
    http://www.nltk.org/api/nltk.html#nltk.text.Text
'''