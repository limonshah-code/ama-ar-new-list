import os
import time
import random
from dotenv import load_dotenv
from paapi5_python_sdk.api.default_api import DefaultApi
from paapi5_python_sdk.models.search_items_request import SearchItemsRequest
from paapi5_python_sdk.models.search_items_resource import SearchItemsResource
from paapi5_python_sdk.models.get_items_resource import GetItemsResource
from paapi5_python_sdk.models.get_items_request import GetItemsRequest
from paapi5_python_sdk.models.partner_type import PartnerType
from paapi5_python_sdk.rest import ApiException
import base64
import json
import requests
import markdown
import mimetypes
import yaml
import re
import smtplib
import csv
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from datetime import datetime, timedelta
from PIL import Image
from google import genai
from google.genai import types
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure output directory
OUTPUT_DIR = "generated-articles"  # Keeping for compatibility but not used
ARTICLE_PROMPTS_DIR = "article-prompts"
KEYWORDS_FILE = "data/keywords.txt"
PROCESSED_KEYWORDS_FILE = "data/processed_keywords.txt"
GENERATED_KEYWORDS_FILE = "data/keywords-generated.txt"
LINKS_FILE = "data/links.txt"
ARTICLES_PER_RUN = 60
TOP_LINKS_COUNT = 1

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTICLE_PROMPTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PROCESSED_KEYWORDS_FILE), exist_ok=True)
os.makedirs(os.path.dirname(GENERATED_KEYWORDS_FILE), exist_ok=True)
os.makedirs(os.path.dirname(LINKS_FILE), exist_ok=True)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load Amazon credentials from .env
load_dotenv()
access_key = os.getenv("AMAZON_ACCESS_KEY")
secret_key = os.getenv("AMAZON_SECRET_KEY")
partner_tag = os.getenv("AMAZON_PARTNER_TAG")

# PAAPI host and region
host = "webservices.amazon.com"
region = "us-east-1"

class APIKeyManager:
    """Manages multiple Gemini API keys with rotation and quota tracking"""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.key_usage_count = {}
        self.max_requests_per_key = 20
        self.failed_keys = set()
        
        for key in self.api_keys:
            self.key_usage_count[key] = 0
    
    def _load_api_keys(self):
        keys = []
        for i in range(1, 7):
            key = os.environ.get(f"GEMINI_API_KEY_{i}")
            if key:
                keys.append(key)
                print(f"Loaded API key #{i}")
        original_key = os.environ.get("GEMINI_API_KEY")
        if original_key and original_key not in keys:
            keys.append(original_key)
            print("Loaded original GEMINI_API_KEY")
        if not keys:
            raise ValueError("No Gemini API keys found in environment variables")
        print(f"Total API keys loaded: {len(keys)}")
        return keys
    
    def get_current_key(self):
        if not self.api_keys:
            raise ValueError("No API keys available")
        current_key = self.api_keys[self.current_key_index]
        if (current_key in self.failed_keys or
            self.key_usage_count[current_key] >= self.max_requests_per_key):
            self._rotate_key()
            current_key = self.api_keys[self.current_key_index]
        return current_key
    
    def _rotate_key(self):
        attempts = 0
        while attempts < len(self.api_keys):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            current_key = self.api_keys[self.current_key_index]
            if (current_key not in self.failed_keys and
                self.key_usage_count[current_key] < self.max_requests_per_key):
                print(f"Rotated to API key #{self.current_key_index + 1}")
                return
            attempts += 1
        raise Exception("All API keys have been exhausted or failed")
    
    def increment_usage(self, key):
        if key in self.key_usage_count:
            self.key_usage_count[key] += 1
            print(f"API key usage: {self.key_usage_count[key]}/{self.max_requests_per_key}")
    
    def mark_key_as_failed(self, key, error_message):
        self.failed_keys.add(key)
        print(f"Marked API key as failed: {error_message}")
        if key == self.api_keys[self.current_key_index]:
            try:
                self._rotate_key()
            except Exception as e:
                print(f"Failed to rotate key: {e}")
    
    def get_status(self):
        status = {}
        for i, key in enumerate(self.api_keys):
            key_id = f"Key_{i+1}"
            status[key_id] = {
                "usage": self.key_usage_count[key],
                "max_requests": self.max_requests_per_key,
                "failed": key in self.failed_keys,
                "active": i == self.current_key_index
            }
        return status

# Initialize API key manager
api_key_manager = APIKeyManager()

def print_separator(title="", length=80):
    if title:
        padding = (length - len(title) - 2) // 2
        print("=" * padding + f" {title} " + "=" * padding)
    else:
        print("=" * length)

def safe_get_value(obj, *keys):
    current = obj
    for key in keys:
        if hasattr(current, key):
            current = getattr(current, key)
        elif isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return "N/A"
    if hasattr(current, 'display_value'):
        return current.display_value
    elif hasattr(current, 'display_values') and current.display_values:
        return current.display_values
    elif current is not None:
        return str(current)
    else:
        return "N/A"

def has_data(obj, *keys):
    current = obj
    for key in keys:
        if hasattr(current, key):
            current = getattr(current, key)
        elif isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return False
    if hasattr(current, 'display_value'):
        return current.display_value and current.display_value != "N/A"
    elif hasattr(current, 'display_values'):
        return current.display_values and len(current.display_values) > 0
    elif current is not None:
        return str(current) != "N/A" and str(current).strip() != ""
    else:
        return False

def rate_limited_delay(min_seconds=2, max_seconds=3):
    delay = random.uniform(min_seconds, max_seconds)
    print(f"   ⏳ Waiting {delay:.1f} seconds to avoid rate limiting...")
    time.sleep(delay)

def fetch_amazon_products(keyword, min_results=7, max_results=12):
    """Fetch Amazon product data and return as a structured list."""
    try:
        default_api = DefaultApi(
            access_key=access_key,
            secret_key=secret_key,
            host=host,
            region=region
        )
        search_resources = [
            SearchItemsResource.ITEMINFO_TITLE,
            SearchItemsResource.ITEMINFO_BYLINEINFO,
            SearchItemsResource.ITEMINFO_CLASSIFICATIONS,
            SearchItemsResource.ITEMINFO_CONTENTINFO,
            SearchItemsResource.ITEMINFO_CONTENTRATING,
            SearchItemsResource.ITEMINFO_EXTERNALIDS,
            SearchItemsResource.ITEMINFO_FEATURES,
            SearchItemsResource.ITEMINFO_MANUFACTUREINFO,
            SearchItemsResource.ITEMINFO_PRODUCTINFO,
            SearchItemsResource.ITEMINFO_TECHNICALINFO,
            SearchItemsResource.ITEMINFO_TRADEININFO,
            SearchItemsResource.IMAGES_PRIMARY_SMALL,
            SearchItemsResource.IMAGES_PRIMARY_MEDIUM,
            SearchItemsResource.IMAGES_PRIMARY_LARGE
        ]
        target_results = random.randint(min_results, max_results)
        request = SearchItemsRequest(
            partner_tag=partner_tag,
            partner_type=PartnerType.ASSOCIATES,
            keywords=keyword,
            resources=search_resources,
            marketplace="www.amazon.com",
            search_index="All",
            item_count=target_results
        )
        print_separator(f"AMAZON PRODUCT SEARCH: '{keyword.upper()}'")
        print(f"🎯 Fetching {target_results} products...\n")
        response = default_api.search_items(request)
        products = []
        if response.search_result is not None and response.search_result.items:
            items = response.search_result.items
            print(f"✅ Found {len(items)} products")
            for idx, item in enumerate(items, start=1):
                product_data = {
                    "asin": item.asin,
                    "title": safe_get_value(item, 'item_info', 'title'),
                    "url": item.detail_page_url,
                    "brand": safe_get_value(item, 'item_info', 'by_line_info', 'brand'),
                    "features": safe_get_value(item, 'item_info', 'features') if has_data(item, 'item_info', 'features') else [],
                    "image_large": safe_get_value(item, 'images', 'primary', 'large', 'url'),
                    "image_medium": safe_get_value(item, 'images', 'primary', 'medium', 'url'),
                    "image_small": safe_get_value(item, 'images', 'primary', 'small', 'url')
                }
                products.append(product_data)
                rate_limited_delay()
        else:
            print("❌ No items found for the keyword:", keyword)
        return products
    except ApiException as e:
        print(f"❌ API Exception: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return []

def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)

def compress_image(image_path, quality=65):
    try:
        with Image.open(image_path) as img:
            webp_path = f"{os.path.splitext(image_path)[0]}.webp"
            img.save(webp_path, 'WEBP', quality=quality)
            os.remove(image_path)
            return webp_path
    except Exception as e:
        print(f"Image compression error: {e}")
        return image_path

def upload_to_cloudinary(file_path, resource_type="image"):
    url = f"https://api.cloudinary.com/v1_1/{os.environ['CLOUDINARY_CLOUD_NAME']}/{resource_type}/upload"
    payload = {
        'upload_preset': 'ml_default',
        'api_key': os.environ['CLOUDINARY_API_KEY']
    }
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, data=payload, files=files)
        if response.status_code == 200:
            return response.json()['secure_url']
        print(f"Upload failed: {response.text}")
        return None
    except Exception as e:
        print(f"Upload error: {e}")
        return None

def generate_and_upload_image(title):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            current_key = api_key_manager.get_current_key()
            client = genai.Client(api_key=current_key)
            model = "gemini-2.0-flash-exp-image-generation"
            contents = [types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"""Create a realistic blog header image for the topic: "{title}". Include the title text "{title}" overlaid on the image as a stylish, readable heading. Use clean typography. Image size should be 16:9 aspect ratio.""")]
            )]
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=["image", "text"])
            )
            api_key_manager.increment_usage(current_key)
            if not response.candidates or not response.candidates[0].content.parts:
                print(f"No valid image data in response (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            inline_data = response.candidates[0].content.parts[0].inline_data
            if not hasattr(inline_data, 'mime_type') or not inline_data.mime_type:
                print(f"Invalid mime_type in response (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            file_ext = mimetypes.guess_extension(inline_data.mime_type)
            if not file_ext:
                print(f"Could not determine file extension (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            original_file = f"blog_image_{int(time.time())}{file_ext}"
            save_binary_file(original_file, inline_data.data)
            final_file = compress_image(original_file)
            image_url = upload_to_cloudinary(final_file)
            if image_url:
                return image_url
            print(f"Image upload failed (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['quota', 'resource_exhausted', 'limit']):
                print(f"Quota exhausted for current API key (attempt {attempt + 1}): {e}")
                api_key_manager.mark_key_as_failed(current_key, str(e))
                if attempt < max_retries - 1:
                    print("Retrying with next API key...")
                    time.sleep(2)
                    continue
            else:
                print(f"Image generation error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
            if attempt == max_retries - 1:
                print("Max retries reached for image generation")
                return None

def create_slug(title):
    slug = title.lower()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'[\s-]+', '-', slug)
    slug = slug.strip('-')
    slug = slug[:100]
    return slug

def link_to_keywords(link):
    url_path = link.split('homeessentialsguide.com/')[-1].strip('/')
    keywords = url_path.replace('-', ' ')
    return keywords

def find_relevant_links(target_keyword, links, top_n=TOP_LINKS_COUNT):
    if not links:
        return []
    link_keywords = [link_to_keywords(link) for link in links]
    all_texts = link_keywords + [target_keyword]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    target_vector = tfidf_matrix[-1]
    link_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(target_vector, link_vectors).flatten()
    link_sim_pairs = list(zip(links, similarities))
    link_sim_pairs.sort(key=lambda x: x[1], reverse=True)
    top_links = [pair[0] for pair in link_sim_pairs[:top_n]]
    print(f"Found {len(top_links)} relevant links for keyword: {target_keyword}")
    return top_links

def extract_category_from_title(title):
    """Extract product category from title for better organization"""
    title_lower = title.lower()
    
    # product category use Reviews as Category
    category_keywords = {
        'cleaning': ['Reviews'],
        'tools': ['Reviews'],
        'kitchen': ['Reviews'],
        'home': ['Reviews'],
        'electronics': ['Reviews'],
        'outdoor': ['Reviews'],
        'fitness': ['Reviews'],
        'automotive': ['Reviews']
    }
    
    for category, keywords in category_keywords.items():
        if any(keyword in title_lower for keyword in keywords):
            return category.capitalize()
    
    return 'Reviews'

def create_enhanced_article_prompt(title, article_number, image_url, products):
    """Enhanced article prompt following the comprehensive template structure"""
    tomorrow = datetime.now() + timedelta(days=1)
    publish_date = tomorrow.strftime("%Y-%m-%dT%H:%M:%SZ")
    slug = create_slug(title)
    canonical_url = f"https://www.homeessentialsguide.com/{slug}"
    
    # Extract category and create relevant tags
    category = extract_category_from_title(title)
    base_keyword = title.lower().replace('best ', '').replace('top ', '')
    
    existing_links = get_existing_links()
    relevant_links = find_relevant_links(title, existing_links)
    
    # Create tags based on title analysis
    tags = [
        base_keyword,
        f"{category} products",
    ]
    
    products_json = json.dumps(products, indent=2)
    
    prompt = f"""Create a comprehensive Amazon product review article in MDX format following this exact structure. Use the provided product data to write detailed, helpful content. The output should be valid MDX with JSX components mixed with must be Markdown format.

---
publishDate: {publish_date}
author: "Mason Everett"
title: "{title}"
excerpt: "Discover the {base_keyword} to enhance your experience. Find top solutions based on thorough testing and reviews."
image: {image_url}
category: "Reviews"
tags:
{chr(10).join(f'  - {tag}' for tag in tags)}
metadata:
  canonical: {canonical_url}
---

import TopPicks from '~/components/ui/TopPicks.astro';
import AmazonProducts from '~/components/ui/AmazonProducts.astro';

## ARTICLE STRUCTURE REQUIREMENTS:

### 1. OPENING SECTION (70-120 words)
Write an engaging introduction in Markdown that:
- Acknowledges the challenge of finding the right {base_keyword}
- Mentions 3-4 key criteria for evaluation (quality, durability, value, performance)
- Introduces the concept of eco-friendly/safe options if applicable
- Promises to help readers make an informed decision
- Use "{title}" naturally in the first 100 words
- Write each sentence inside its own <p> tag

### 2. TOP PICKS SECTION
Next, include the <TopPicks /> component.
- id="top-picks"
- tagline="Expert Recommendations ✨"
- title="Our Top Picks"
- subtitle="After testing dozens of products, we've selected the best options for every budget and need. Each recommendation has been thoroughly evaluated for performance, durability, and value."
- items: An array including all products from the data, assigning appropriate categories like 'Best Overall', 'Best Budget', 'Best Premium', 'Best Multi-functional', creating custom categories as needed for all products.
  - category: e.g., 'Best Overall', 'Best Budget', etc.
  - title: Shortened and engaging version of the product title from data (keep it concise and short)
  - description: 2-3 sentences explaining why it's picked, based on features. Do not use any double quotes in the description.
  - link: Product URL from data
  - image: Product image_large or image_medium from data
  - color: Cycle through 'indigo', 'teal', 'green', 'purple'

Use actual product data to fill these.

### 3. PRODUCT REVIEWS SECTION
Title: "## {len(products)} {title}"

write a short, engaging, and persuasive description (2–3 sentences) that hooks the reader.  
- Mention why these products matter.  
- Create a sense of excitement or curiosity to keep reading.  

For EACH product in the data, include an <AmazonProducts /> component. Increment cardNumber starting from 1.
For each component:
- cardNumber: The product number (1,2,3...)
- title: Shortened and engaging version of the full product title from data (keep it concise and short)
- subtitle: A short tagline like Best in class for [key benefit] generated based on features
- image: Product image_large from data
- amazonLink: Product URL from data
- description: 100-150 word paragraph covering main benefits, standout features, performance, who it's for. Do not use any double quotes in the description.
- whySelected: 50-100 words on why we selected it, mentioning testing. Do not use any double quotes.
- userExperience: 50-100 words on typical user experiences, praises, and uses. Do not use any double quotes.
- features: Array of 4-7 strings based on product features from data, with benefits
- pros: Array of 3-5 strings, positive aspects
- cons: Array of 2-3 strings, minor drawbacks
- specs: Array of objects like {{ label: "Feature Name", value: "Value" }} - Generate 3-6 plausible specs from features or infer

Use actual data where possible, generate content based on it.

### 4. BUYING GUIDE SECTION
Title: "## What to Look for Before Buying the {title}"

Include these subsections with detailed explanations in Markdown:
1. ### Find Your Need - Compatibility and use case considerations
2. ### Budget - Setting realistic price expectations ($X-$Y ranges)
3. ### Key Features - Most important technical specifications
4. ### Quality & Durability - What to look for in construction
5. ### Brand Reputation - Importance of established manufacturers
6. ### Portability/Size - Considerations for space and storage

### 5. EDUCATIONAL CONTENT

#### Value Proposition Section
Title: "## Is it A Wise Decision to Buy {base_keyword}?"
- Justify the investment (150-200 words) in Markdown
- Compare to alternatives
- Highlight long-term benefits
- Address cost concerns

#### Best Practices Section
Title: "## What Is the Best Way to Choose {base_keyword}?"
- Step-by-step selection guide in Markdown
- Professional tips
- Common mistakes to avoid
- When to upgrade vs. budget options
- add more option if available 

### 6. FAQ SECTION
Title: "## FAQ"

Create 8-10 FAQs in Markdown covering:
- "What is the best {base_keyword} for beginners?"
- "How much should I spend on {base_keyword}?"
- "What features are most important in {base_keyword}?"
- "How do I maintain my {base_keyword}?"
- "Are expensive {base_keyword} worth it?"
- "What brands make the best {base_keyword}?"
- "How long do {base_keyword} typically last?"
- "Can I find good {base_keyword} on a budget?"

Each answer should be 50-75 words with actionable advice.

### 7. FINAL VERDICT
Title: "## Final Verdict"

- Restate top recommendation with key selling points in Markdown
- Mention budget option for cost-conscious buyers
- Highlight premium option for quality seekers
- End with clear, actionable next steps
- Product recommendation with product URL (Must Follow this)
- Write each sentence inside its own <p> tag

## WRITING GUIDELINES:

**Tone & Voice:**
- Conversational and helpful
- Authoritative but approachable: Use confident language like "we research" and "we recommend" to establish credibility without being overly formal.
- Address reader directly with "you"

**Content Principles:**
- Lead with benefits, support with features
- Use specific details from the product data
- Include safety considerations
- Balance enthusiasm with honest assessment
- Critical Formatting Rule: No Double Quotes, The final output for both the title and the excerpt must not contain any double quotes (").

**SEO Structure:**
- Use H2/H3 headings properly
- Write scannable content with bullets and short paragraphs
- Target 4,000-5,000 words total

## AMAZON PRODUCT DATA TO USE:
{products_json}

**CRITICAL INSTRUCTIONS:**
1. Use product titles, features, images, and URLs from the data above, but shorten titles to be short and engaging
2. Every product component must include its Amazon affiliate link
3. Write unique content for each product based on its specific features
4. Maintain consistent quality and structure across all articles
5. Ensure all product claims are based on the provided data
6. Create compelling reasons to buy each product
7. Include proper MDX formatting throughout, with JSX for components and Markdown for text.
Generate and use Seo Title in the article must:
- Keyword: Must include the primary keyword (e.g., "best steam cleaner for bathroom grout").
- Format: Incorporate a number (e.g., "10 Best") and/or the current year to signal freshness and comprehensiveness.
- Tone: The title should be clear, informative, and engaging.
- Length: Aim for 50-60 characters.
- Constraint: Do not use any double quotes and colon also.

Generate and use Seo Excerpt in the article must:
- Purpose: Function as a meta description designed to attract clicks from search engine results pages.
- Keyword: Must include the primary keyword.
- Structure:
  - Hook: Start with a question that addresses the reader's pain point directly.
  - Value Proposition: State your authority and what you did.
  - Benefit: Promise a clear, desirable outcome.
  - Tone: Must be conversational and helpful, directly addressing the reader with "you".
  - Length: Keep it concise and impactful, between 140-150 characters.
  - Constraint: Do not use any double quotes and colon also.
Generate the complete MDX article now, following this structure exactly and using the provided product data comprehensively."""

    return prompt

def count_generated_prompts():
    """Count the total number of prompt files in the article-prompts directory"""
    try:
        if os.path.exists(ARTICLE_PROMPTS_DIR):
            files = [f for f in os.listdir(ARTICLE_PROMPTS_DIR) 
                    if os.path.isfile(os.path.join(ARTICLE_PROMPTS_DIR, f)) and f.endswith('.txt')]
            return len(files)
        return 0
    except Exception as e:
        print(f"Error counting prompt files: {e}")
        return 0

def send_email_notification(titles, article_urls, prompt_lengths, remaining_keywords, total_prompts, recipient_email="limonseosolution@gmail.com"):
    from_email = "limon.working@gmail.com"
    app_password = os.environ.get("EMAIL_PASSWORD")
    if not app_password:
        print("Email password not set. Skipping email notification.")
        return False
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = recipient_email
    msg['Subject'] = f"Generated Image and Article Prompts New- {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    api_status = api_key_manager.get_status()
    status_text = "\n\nAPI Key Usage Status:\n"
    for key_id, status in api_status.items():
        status_text += f"{key_id}: {status['usage']}/{status['max_requests']} requests"
        if status['failed']:
            status_text += " (FAILED)"
        if status['active']:
            status_text += " (ACTIVE)"
        status_text += "\n"
    
    # Calculate prompt statistics
    avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
    min_prompt_length = min(prompt_lengths) if prompt_lengths else 0
    max_prompt_length = max(prompt_lengths) if prompt_lengths else 0
    
    body = f"""Image and Article Prompt Generation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 SUMMARY:
- Successfully processed: {len(titles)} keywords
- Remaining keywords in file: {remaining_keywords}
- Total prompts in folder: {total_prompts}
- Average prompt length: {avg_prompt_length:.0f} characters
- Min prompt length: {min_prompt_length} characters
- Max prompt length: {max_prompt_length} characters

📝 PROCESSED ITEMS:
"""
    for i, (title, url, prompt_len) in enumerate(zip(titles, article_urls, prompt_lengths), 1):
        body += f"{i}. {title} ({prompt_len} chars)\n   URL: {url}\n\n"
    
    body += status_text
    
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, app_password)
        server.send_message(msg)
        server.quit()
        print(f"Email notification sent successfully to {recipient_email}")
        return True
    except Exception as e:
        print(f"Failed to send email notification: {e}")
        return False

def read_keywords_from_csv(filename=KEYWORDS_FILE):
    try:
        keywords = []
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and row[0].strip():
                        keywords.append(row[0].strip())
        else:
            print(f"Keywords file {filename} not found. Creating new file.")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', encoding='utf-8'):
                pass
        return keywords
    except Exception as e:
        print(f"Error reading keywords from CSV: {e}")
        return []

def write_keywords_to_csv(keywords, filename=KEYWORDS_FILE):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for keyword in keywords:
                writer.writerow([keyword])
        return True
    except Exception as e:
        print(f"Error writing keywords to CSV: {e}")
        return False

def append_processed_keywords(keywords, urls, filename=PROCESSED_KEYWORDS_FILE):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
        with open(filename, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if not file_exists:
                f.write("# Processed Keywords Log\n")
                f.write("# Format: [TIMESTAMP] KEYWORD - URL\n\n")
            f.write(f"## Batch processed on {timestamp}\n")
            for keyword, url in zip(keywords, urls):
                f.write(f"{url}\n")
            f.write("\n")
        return True
    except Exception as e:
        print(f"Error appending to processed keywords file: {e}")
        return False

def append_to_generated_keywords(keywords, filename=GENERATED_KEYWORDS_FILE):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for keyword in keywords:
                f.write(f"{keyword}\n")
        return True
    except Exception as e:
        print(f"Error appending to generated keywords file: {e}")
        return False

def append_to_links_file(old_urls, new_urls, filename=LINKS_FILE):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        existing_links = set()
        if os.path.exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, 'r', encoding='utf-8') as f:
                existing_links = {line.strip() for line in f if line.strip()}
        all_urls = old_urls + new_urls
        unique_new_urls = [url for url in all_urls if url not in existing_links]
        if unique_new_urls:
            with open(filename, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for url in unique_new_urls:
                    f.write(f"{url}\n")
                f.write("\n")
            return True
        else:
            print("No new unique URLs to append")
            return True
    except Exception as e:
        print(f"Error appending to links file: {e}")
        return False

def get_existing_links(filename=LINKS_FILE):
    existing_links = []
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and 'https://' in line:
                    existing_links.append(line)
    return existing_links

def get_keywords(filename=KEYWORDS_FILE, count=ARTICLES_PER_RUN):
    try:
        all_keywords = read_keywords_from_csv(filename)
        if not all_keywords:
            print("No keywords found in the CSV file or file is empty.")
            return [" "], []
        if len(all_keywords) <= count:
            selected_keywords = all_keywords.copy()
            return selected_keywords, selected_keywords
        last_index = 0
        track_file = ".last_keyword_index"
        if os.path.exists(track_file):
            with open(track_file, 'r') as f:
                try:
                    last_index = int(f.read().strip())
                except ValueError:
                    last_index = 0
        start_index = last_index % len(all_keywords)
        end_index = start_index + count
        if end_index <= len(all_keywords):
            selected_keywords = all_keywords[start_index:end_index]
        else:
            selected_keywords = all_keywords[start_index:] + all_keywords[:end_index - len(all_keywords)]
        with open(track_file, 'w') as f:
            f.write('0')
        return selected_keywords, selected_keywords
    except Exception as e:
        print(f"Error reading keywords file: {e}")
        return [" "], []

def update_keyword_files(all_used_keywords, article_urls):
    if not all_used_keywords:
        print("No keywords to update.")
        return
    all_keywords = read_keywords_from_csv(KEYWORDS_FILE)
    remaining_keywords = [k for k in all_keywords if k not in all_used_keywords]
    if write_keywords_to_csv(remaining_keywords, KEYWORDS_FILE):
        print(f"Removed {len(all_used_keywords)} used keywords from Text file.")
    else:
        print("Failed to update CSV file with remaining keywords.")
    if append_processed_keywords(all_used_keywords, article_urls, PROCESSED_KEYWORDS_FILE):
        print(f"Added {len(all_used_keywords)} keywords to processed keywords file with URLs.")
    else:
        print("Failed to update processed keywords file.")
    if append_to_generated_keywords(all_used_keywords, GENERATED_KEYWORDS_FILE):
        print(f"Added {len(all_used_keywords)} keywords to generated keywords file.")
    else:
        print("Failed to update generated keywords file.")
    existing_links = get_existing_links(LINKS_FILE)
    old_links = ["https://www.homeessentialsguide.com/how-to-clean-a-ceiling"]
    filtered_old_links = [link for link in old_links if link not in existing_links]
    filtered_new_links = [link for link in article_urls if link not in existing_links]
    if filtered_old_links or filtered_new_links:
        if append_to_links_file(filtered_old_links, filtered_new_links, LINKS_FILE):
            print(f"Added {len(filtered_old_links)} old links and {len(filtered_new_links)} new links to links file.")
        else:
            print("Failed to update links file.")

def main():
    print(f"🚀 Starting Image Generation and Article Prompt Creation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator("INITIALIZATION")
    
    print(f"API Key Manager Status:")
    for key_id, status in api_key_manager.get_status().items():
        print(f"  {key_id}: {'Active' if status['active'] else 'Standby'}")
    
    keywords, keywords_to_track = get_keywords()
    print(f"\n📋 Selected {len(keywords)} keywords for processing:")
    for i, keyword in enumerate(keywords, 1):
        print(f"  {i}. {keyword}")
    
    successful_keywords = []
    article_urls = []
    prompt_lengths = []
    default_image_url = "https://res.cloudinary.com/dbcpfy04c/image/upload/v1743184673/images_k6zam3.png"
    
    for i, title in enumerate(keywords, 1):
        print_separator(f"PROCESSING ITEM #{i}: {title.upper()}")
        
        try:
            # Create URL slug and article URL
            slug = create_slug(title)
            article_url = f"https://www.homeessentialsguide.com/{slug}"
            print(f"📝 Article URL: {article_url}")
            
            # Generate and upload header image
            print("\n🎨 Generating header image...")
            image_url = generate_and_upload_image(title)
            if not image_url:
                print(f"⚠️ Using default image URL: {default_image_url}")
                image_url = default_image_url
            else:
                print(f"✅ Generated image URL: {image_url}")
            
            # Fetch Amazon products
            print(f"\n🛒 Fetching Amazon products for: {title}")
            products = fetch_amazon_products(title)
            if not products:
                print("❌ No products fetched, skipping to next keyword")
                continue
            
            print(f"✅ Successfully fetched {len(products)} products")
            
            # Create enhanced article prompt
            print("\n✍️ Creating article prompt...")
            prompt = create_enhanced_article_prompt(title, i, image_url, products)
            prompt_length = len(prompt)
            prompt_lengths.append(prompt_length)
            print(f"📊 Prompt length: {prompt_length} characters")
            
            # Save prompt to file
            prompt_filename = f"{ARTICLE_PROMPTS_DIR}/{title.lower().replace(' ', '-')}.txt"
            try:
                os.makedirs(os.path.dirname(prompt_filename), exist_ok=True)
                with open(prompt_filename, "w", encoding="utf-8") as f:
                    f.write(prompt)
                print(f"✅ Prompt saved to {prompt_filename}")
            except Exception as e:
                print(f"❌ Error saving prompt to {prompt_filename}: {e}")
            
            successful_keywords.append(title)
            article_urls.append(article_url)
            
        except Exception as e:
            print(f"❌ Error processing keyword '{title}': {e}")
            continue
        
        # Rate limiting between items
        if i < len(keywords):
            print("\n⏳ Waiting 10 seconds before next item...")
            time.sleep(10)
    
    # Get remaining keywords count
    all_keywords = read_keywords_from_csv(KEYWORDS_FILE)
    remaining_keywords_count = len(all_keywords) - len(successful_keywords)
    
    # Count total prompts in folder
    total_prompts_count = count_generated_prompts()
    
    # Final reporting
    print_separator("FINAL RESULTS")
    
    print(f"📈 Final API Key Usage Status:")
    for key_id, status in api_key_manager.get_status().items():
        print(f"  {key_id}: {status['usage']}/{status['max_requests']} requests"
              f"{' (FAILED)' if status['failed'] else ''}")
    
    # Update keyword tracking files
    print("\n📁 Updating keyword files...")
    update_keyword_files(successful_keywords, article_urls)
    
    # Send email notification
    if successful_keywords:
        print("\n📧 Sending email notification...")
        send_email_notification(
            successful_keywords, 
            article_urls, 
            prompt_lengths,
            remaining_keywords_count,
            total_prompts_count
        )
    
    # Final summary
    print_separator("GENERATION COMPLETE")
    print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Successfully processed: {len(successful_keywords)} keywords")
    print(f"📊 Success rate: {(len(successful_keywords)/len(keywords)*100):.1f}%")
    print(f"📈 Remaining keywords in file: {remaining_keywords_count}")
    print(f"📁 Total prompts in folder: {total_prompts_count}")
    
    if prompt_lengths:
        print(f"📝 Prompt length stats:")
        print(f"   Average: {sum(prompt_lengths)/len(prompt_lengths):.0f} characters")
        print(f"   Min: {min(prompt_lengths)} characters")
        print(f"   Max: {max(prompt_lengths)} characters")
    
    print(f"\n📋 Processed Items:")
    for i, (keyword, url, prompt_len) in enumerate(zip(successful_keywords, article_urls, prompt_lengths), 1):
        print(f"  {i}. {keyword} ({prompt_len} chars)")
        print(f"     → {url}")
    
    if len(successful_keywords) < len(keywords):
        failed_keywords = [k for k in keywords if k not in successful_keywords]
        print(f"\n⚠️ Failed Keywords ({len(failed_keywords)}):")
        for keyword in failed_keywords:
            print(f"  - {keyword}")

if __name__ == "__main__":
    main()