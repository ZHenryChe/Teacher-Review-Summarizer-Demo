import requests
from bs4 import BeautifulSoup

def scraping_baby_txt(urls: list) -> dict:

    '''
    Given a list of urls from RateMyProfessor, returns review
    texts only from each url.
    '''

    review_compile = () #tuple to store text

    for url in urls:

        #Use request library to get url's HTML
        data = requests.get(url)

        #Makes the url into a readable file
        soup = BeautifulSoup(data.text, 'lxml')

        #Finding all review windows
        reviews = soup.find_all('div', class_ = "Rating__StyledRating-sc-1rhvpxz-1 jcIQzP")

        for review in reviews:

            #Find the text portion of review
            review_text = review.find('div', class_ = "Comments__StyledComments-dzzyvm-0 gRjWel").text

            #adds dictionary to list compilation
            review_compile += (review_text,)

    return review_compile

def txt_file_maker_from_tuple(info: tuple, **kwargs) -> None:

    '''
    Makes a txt_file from values within a tuple. Each value will be
    separated by two \n characters. File name is defaulted to
    "TEST.txt".
    '''

    new_file = open(kwargs.get('file_name', "TEST.txt"), 'w')

    for item in info:

        new_file.write(item)
        new_file.write('\n\n')

    new_file.close()


