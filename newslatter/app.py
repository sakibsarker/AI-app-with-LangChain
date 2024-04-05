import streamlit as st
from helpers import *


def main():
    query="Google"
    resp=search_serp(query)
    urls=pick_best_articles_urls(response_json=resp,query=query)
    data=extract_content_from_urls(urls=urls)
    summaries=summarizer(db=data,query=query)
    print(summaries)
    pass

if __name__=='__main__':
    main()