import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import quote
from typing import List

# Default configuration
SEARCH_KEYWORDS = [
    "modular design",
    "software architecture",
    "system design",
    "industrial design",
    "supply chain"
]

DATA_DIR = "web_sources"

def search_wikipedia(keyword):
    """Search Wikipedia for a keyword and return the first result URL"""
    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(keyword)}&format=json"
    response = requests.get(search_url)
    data = response.json()
    
    if data['query']['search']:
        title = data['query']['search'][0]['title']
        return f"https://en.wikipedia.org/wiki/{quote(title)}"
    return None

def scrape_wikipedia_article(keyword):
    """Scrape a Wikipedia article and return its content"""
    url = search_wikipedia(keyword)
    if not url:
        return None
    
    try:
        # Fetch the article
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get the main content
        content = soup.find('div', {'class': 'mw-parser-output'})
        if not content:
            return None
        
        # Extract text from paragraphs and headers
        article_text = ""
        for element in content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = element.get_text().strip()
            if text:
                article_text += text + "\n\n"
        
        return article_text
        
    except Exception as e:
        print(f"❌  Error fetching {keyword}: {str(e)}")
        return None

def scrape_and_save_article(url, keyword):
    """Scrape a Wikipedia article and save it as a text file"""
    print(f"▶️  Searching for: {keyword}")
    
    try:
        # Fetch the article
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get the main content
        content = soup.find('div', {'class': 'mw-parser-output'})
        if not content:
            print(f"❌  Could not find article content for: {keyword}")
            return
        
        # Extract text from paragraphs and headers
        article_text = ""
        for element in content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = element.get_text().strip()
            if text:
                article_text += text + "\n\n"
        
        # Create output directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Save to file
        filename = f"wikipedia_{keyword.replace(' ', '_')}.txt"
        filepath = os.path.join(DATA_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(article_text)
            
        print(f"✅  Saved article to: {filepath}")
        
    except Exception as e:
        print(f"❌  Error processing {keyword}: {str(e)}")

def scrape_wikipedia_articles(keywords: List[str] = None) -> None:
    """Main function to scrape Wikipedia articles for given keywords"""
    if keywords is None:
        keywords = SEARCH_KEYWORDS
    
    # Create output directory
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"🔍 Starting Wikipedia article search for {len(keywords)} keywords...")
    
    for keyword in keywords:
        print(f"▶️  Searching for: {keyword}")
        try:
            # Clean the keyword for filename
            clean_keyword = keyword.replace(" ", "_").lower()
            filename = f"wikipedia_{clean_keyword}.txt"
            filepath = os.path.join(DATA_DIR, filename)
            
            content = scrape_wikipedia_article(keyword)
            if content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅  Saved article to: {filepath}")
            else:
                print(f"❌  Failed to retrieve content for: {keyword}")
                
        except Exception as e:
            print(f"❌  Error processing {keyword}: {str(e)}")
    
    print("\n✨ Article scraping complete!")

def main():
    """Example usage of the Wikipedia scraper."""
    scrape_wikipedia_articles()

if __name__ == "__main__":
    main() 