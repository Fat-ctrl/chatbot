import requests
import markdownify
import re
import os
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

MAX_ARTICLES = 200  
save_location = "articles/"

def main():
    # Create output folder if it doesn't exist
    os.makedirs(save_location, exist_ok=True)

    # API endpoint
    base_url = os.getenv("API_URL")
    if not base_url:
        raise ValueError("API_URL environment variable is not set. Please configure it in your environment.")
    articles = []

    # Fetch up to MAX_ARTICLES with pagination
    next_url = base_url
    while len(articles) < MAX_ARTICLES and next_url:
        try:
            response = requests.get(next_url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching articles: {e}")
            break
        
        articles.extend(data.get("articles", []))
        next_url = data.get("next_page")

    # Limit to MAX_ARTICLES
    articles_to_save = articles[:MAX_ARTICLES]

    # Convert and save each article
    for article in articles_to_save:
        title_text = article['title']
        url = article['html_url']
        
        # Extract slug from html_url
        path = urlparse(url).path 
        slug_match = re.search(r'/articles/\d+-([a-zA-Z0-9\-]+)', path)
        slug = slug_match.group(1) if slug_match else "untitled"

        filename = f"{slug}.md"

        markdown_title = f"# {title_text}\n"
        url_md = f"[View Article]({url})\n"
        body_md = markdownify.markdownify(article.get("body", ""), heading_style="ATX")
        full_md = f"{markdown_title}\n{url_md}\n\n{body_md}\n"

        with open(os.path.join(save_location, filename), "w", encoding="utf-8") as f:
            f.write(full_md)

    print(f"Saved {len(articles_to_save)} articles using slugs from URL.")

if __name__ == "__main__":
    main()
