#!/usr/bin/env python3
"""
PyNucleus Wikipedia Scraper
Automated knowledge base expansion through Wikipedia content collection
"""

import sys
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional, Set
from urllib.parse import quote_plus
from loguru import logger
import re

# Add src to path for enhanced text cleaner import
root_dir = Path(__file__).parent.parent.parent.parent
scripts_path = str(root_dir / "scripts")
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

# Import enhanced text cleaner
from enhanced_text_cleaner import EnhancedTextCleaner

# Add Wikipedia to requirements if not present
try:
    import wikipedia
except ImportError:
    logger.warning("Wikipedia package not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wikipedia-api"])
    import wikipedia

class WikipediaScraper:
    """Scrapes Wikipedia articles related to chemical engineering topics."""
    
    def __init__(self, output_dir: str = "data/01_raw/wikipedia"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize text cleaner
        self.text_cleaner = EnhancedTextCleaner()
        
        # Chemical engineering keywords for targeted scraping
        self.keywords = [
            # Core Chemical Engineering
            "chemical engineering", "chemical process", "unit operations",
            "mass transfer", "heat transfer", "momentum transfer",
            "thermodynamics", "kinetics", "reactor design",
            
            # Process Design & Optimization
            "process design", "process optimization", "process simulation",
            "distillation", "absorption", "extraction", "crystallization",
            "filtration", "separation processes", "membrane technology",
            
            # Equipment & Systems
            "heat exchanger", "reactor design", "distillation column",
            "pump", "compressor", "turbine", "pressure vessel",
            "piping system", "instrumentation", "control systems",
            
            # Safety & Environment
            "process safety", "chemical safety", "hazard analysis",
            "environmental engineering", "pollution control",
            "waste treatment", "sustainability",
            
            # Advanced Topics
            "process intensification", "green chemistry", "biotechnology",
            "nanotechnology", "catalysis", "electrochemistry",
            "materials science", "polymer engineering"
        ]
        
        # Processed articles tracking
        self.processed_titles: Set[str] = set()
        self.failed_articles: List[str] = []
        
        # Configure Wikipedia
        wikipedia.set_lang("en")
        wikipedia.set_rate_limiting(True)
        
    def scrape_keywords(self, max_articles_per_keyword: int = 3) -> Dict[str, int]:
        """
        Scrape Wikipedia articles for all chemical engineering keywords.
        
        Args:
            max_articles_per_keyword: Maximum articles to collect per keyword
            
        Returns:
            Dict with scraping statistics
        """
        stats = {
            "keywords_processed": 0,
            "articles_collected": 0,
            "articles_failed": 0,
            "total_text_size": 0
        }
        
        logger.info(f"Starting Wikipedia scraping for {len(self.keywords)} keywords")
        
        for keyword in self.keywords:
            logger.info(f"Processing keyword: '{keyword}'")
            
            try:
                # Search for articles related to keyword
                search_results = wikipedia.search(keyword, results=max_articles_per_keyword * 2)
                
                articles_collected = 0
                for title in search_results:
                    if articles_collected >= max_articles_per_keyword:
                        break
                        
                    if title.lower() in self.processed_titles:
                        logger.debug(f"Skipping already processed: {title}")
                        continue
                    
                    # Check if title is relevant to chemical engineering
                    if not self._is_relevant_article(title):
                        logger.debug(f"Skipping irrelevant article: {title}")
                        continue
                    
                    # Scrape the article
                    article_data = self._scrape_article(title)
                    if article_data:
                        # Save article
                        self._save_article(article_data, keyword)
                        articles_collected += 1
                        stats["articles_collected"] += 1
                        stats["total_text_size"] += len(article_data["content"])
                        
                        # Add delay to respect rate limits
                        time.sleep(1)
                    else:
                        stats["articles_failed"] += 1
                
                stats["keywords_processed"] += 1
                logger.info(f"Collected {articles_collected} articles for '{keyword}'")
                
            except Exception as e:
                logger.error(f"Error processing keyword '{keyword}': {e}")
                stats["articles_failed"] += 1
        
        # Save scraping summary
        self._save_scraping_summary(stats)
        
        logger.info(f"Scraping completed: {stats['articles_collected']} articles collected")
        return stats
    
    def _is_relevant_article(self, title: str) -> bool:
        """Check if article title is relevant to chemical engineering."""
        title_lower = title.lower()
        
        # Chemical engineering related terms
        relevant_terms = [
            "chemical", "process", "engineering", "reactor", "distillation",
            "separation", "heat", "mass", "transfer", "thermodynamics",
            "catalyst", "industrial", "manufacturing", "plant", "equipment",
            "safety", "environmental", "pollution", "treatment", "membrane",
            "crystallization", "absorption", "extraction", "filtration"
        ]
        
        # Check if title contains relevant terms
        return any(term in title_lower for term in relevant_terms)
    
    def _scrape_article(self, title: str) -> Optional[Dict]:
        """
        Scrape a single Wikipedia article.
        
        Args:
            title: Wikipedia article title
            
        Returns:
            Article data dictionary or None if failed
        """
        try:
            logger.debug(f"Scraping article: {title}")
            
            # Get Wikipedia page
            page = wikipedia.page(title)
            
            # Clean and process content using enhanced text cleaner
            original_content = page.content
            cleaned_content = self._clean_wikipedia_content(original_content)
            cleaned_summary = self._clean_wikipedia_content(page.summary)
            
            if len(cleaned_content) < 500:  # Skip very short articles
                logger.debug(f"Skipping short article: {title} ({len(cleaned_content)} chars)")
                return None
            
            article_data = {
                "title": page.title,
                "url": page.url,
                "content": cleaned_content,
                "summary": cleaned_summary,
                "categories": getattr(page, 'categories', [])[:10],  # Limit categories
                "references": getattr(page, 'references', [])[:10],  # Limit references
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "word_count": len(cleaned_content.split()),
                "char_count": len(cleaned_content),
                "original_length": len(original_content),
                "cleaning_efficiency": f"{len(cleaned_content)/len(original_content)*100:.1f}%"
            }
            
            self.processed_titles.add(title.lower())
            return article_data
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Try first disambiguation option
            try:
                logger.debug(f"Disambiguation for {title}, trying: {e.options[0]}")
                return self._scrape_article(e.options[0])
            except:
                logger.warning(f"Failed to resolve disambiguation for: {title}")
                return None
                
        except wikipedia.exceptions.PageError:
            logger.warning(f"Page not found: {title}")
            return None
            
        except Exception as e:
            logger.error(f"Error scraping {title}: {e}")
            self.failed_articles.append(title)
            return None
    
    def _clean_wikipedia_content(self, content: str) -> str:
        """Clean Wikipedia content using our enhanced text cleaner."""
        
        # First pass: Remove Wikipedia-specific patterns
        patterns_to_remove = [
            r'\[\d+\]',  # Reference numbers [1], [2], etc.
            r'=+ .* =+\n',  # Section headers like === Section ===
            r'See also\n.*?(?=\n\n|\n[A-Z]|\Z)',  # See also sections
            r'References\n.*?(?=\n\n|\n[A-Z]|\Z)',  # References sections  
            r'External links\n.*?(?=\n\n|\n[A-Z]|\Z)',  # External links sections
            r'Categories:.*$',  # Categories
            r'{{.*?}}',  # Template markup
            r'\[\[File:.*?\]\]',  # File references
            r'\[\[Image:.*?\]\]',  # Image references
            r'\[\[Category:.*?\]\]',  # Category links
        ]
        
        cleaned = content
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.DOTALL)
        
        # Second pass: Use enhanced text cleaner for deep cleaning
        cleaned = self.text_cleaner.clean_text(cleaned)
        
        # Final pass: Wikipedia-specific post-processing
        # Remove remaining wiki markup
        cleaned = re.sub(r'\[\[(.*?)\|(.*?)\]\]', r'\2', cleaned)  # [[link|text]] -> text
        cleaned = re.sub(r'\[\[(.*?)\]\]', r'\1', cleaned)  # [[link]] -> link
        cleaned = re.sub(r"'''(.*?)'''", r'\1', cleaned)  # Bold markup
        cleaned = re.sub(r"''(.*?)''", r'\1', cleaned)  # Italic markup
        
        # Clean up excessive whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _save_article(self, article_data: Dict, keyword: str):
        """Save article data to both TXT and JSON files with comprehensive content."""
        
        # Create safe filename
        safe_title = re.sub(r'[^\w\-_.]', '_', article_data["title"])
        safe_title = safe_title[:100]  # Limit filename length
        
        # Create keyword subdirectory
        keyword_dir = self.output_dir / keyword.replace(" ", "_")
        keyword_dir.mkdir(exist_ok=True)
        
        # Save comprehensive text file with full content
        txt_file = keyword_dir / f"{safe_title}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            # Header with metadata
            f.write(f"Title: {article_data['title']}\n")
            f.write(f"Source: Wikipedia - {article_data['url']}\n")
            f.write(f"Scraped: {article_data['scraped_at']}\n")
            f.write(f"Word Count: {article_data['word_count']}\n")
            f.write(f"Character Count: {article_data['char_count']}\n")
            f.write(f"Keywords: {keyword}\n")
            f.write("-" * 80 + "\n\n")
            
            # Summary section
            f.write("SUMMARY:\n")
            f.write(article_data['summary'])
            f.write("\n\n" + "=" * 80 + "\n\n")
            
            # Main content (cleaned)
            f.write("CONTENT:\n")
            f.write(article_data['content'])
            
            # Additional metadata if available
            if article_data.get('categories'):
                f.write("\n\n" + "-" * 40 + "\n")
                f.write("CATEGORIES:\n")
                for category in article_data['categories']:
                    f.write(f"• {category}\n")
        
        # Save comprehensive JSON metadata
        json_file = keyword_dir / f"{safe_title}_metadata.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            # Add cleaning statistics
            article_data['text_cleaning_stats'] = self.text_cleaner.stats.copy()
            article_data['keyword'] = keyword
            article_data['files_created'] = {
                'txt_file': str(txt_file.relative_to(self.output_dir.parent)),
                'json_file': str(json_file.relative_to(self.output_dir.parent))
            }
            
            json.dump(article_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved article: {article_data['title']}")
        logger.debug(f"  TXT: {txt_file}")
        logger.debug(f"  JSON: {json_file}")
        logger.debug(f"  Cleaned content: {len(article_data['content'])} chars")
    
    def _save_scraping_summary(self, stats: Dict):
        """Save comprehensive scraping session summary."""
        summary_file = self.output_dir / "scraping_summary.json"
        
        summary_data = {
            "scraping_session": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "statistics": stats,
                "keywords_used": self.keywords,
                "failed_articles": self.failed_articles,
                "total_processed_titles": len(self.processed_titles),
                "text_cleaning_summary": self.text_cleaner.stats.copy()
            }
        }
        
        # Add file structure information
        summary_data["output_structure"] = {
            "base_directory": str(self.output_dir),
            "keyword_directories": [],
            "total_files_created": {"txt": 0, "json": 0}
        }
        
        # Count files in each keyword directory
        for keyword_dir in self.output_dir.iterdir():
            if keyword_dir.is_dir():
                txt_files = list(keyword_dir.glob("*.txt"))
                json_files = list(keyword_dir.glob("*.json"))
                summary_data["output_structure"]["keyword_directories"].append({
                    "keyword": keyword_dir.name,
                    "txt_files": len(txt_files),
                    "json_files": len(json_files)
                })
                summary_data["output_structure"]["total_files_created"]["txt"] += len(txt_files)
                summary_data["output_structure"]["total_files_created"]["json"] += len(json_files)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Scraping summary saved to: {summary_file}")
        logger.info(f"Text cleaning stats: {self.text_cleaner.stats}")

def scrape_sample_data():
    """Scrape a small sample of Wikipedia data for testing."""
    scraper = WikipediaScraper()
    
    # Test with a few key topics
    test_keywords = [
        "distillation", "heat exchanger", "chemical reactor", 
        "mass transfer", "process safety"
    ]
    
    scraper.keywords = test_keywords
    stats = scraper.scrape_keywords(max_articles_per_keyword=2)
    
    return stats

def main():
    """Main entry point for Wikipedia scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PyNucleus Wikipedia Scraper")
    parser.add_argument("--output-dir", "-o", default="data/01_raw/wikipedia",
                        help="Output directory for scraped content")
    parser.add_argument("--max-articles", "-n", type=int, default=3,
                        help="Maximum articles per keyword")
    parser.add_argument("--sample", action="store_true",
                        help="Run with sample data only")
    parser.add_argument("--keywords", nargs="+",
                        help="Custom keywords to scrape")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    
    scraper = WikipediaScraper(output_dir=args.output_dir)
    
    if args.sample:
        logger.info("Running sample Wikipedia scraping...")
        stats = scrape_sample_data()
    else:
        if args.keywords:
            scraper.keywords = args.keywords
        
        logger.info(f"Starting Wikipedia scraping to: {args.output_dir}")
        stats = scraper.scrape_keywords(max_articles_per_keyword=args.max_articles)
    
    logger.info("Wikipedia scraping completed!")
    logger.info(f"Articles collected: {stats['articles_collected']}")
    logger.info(f"Total text size: {stats['total_text_size']:,} characters")

if __name__ == "__main__":
    main() 