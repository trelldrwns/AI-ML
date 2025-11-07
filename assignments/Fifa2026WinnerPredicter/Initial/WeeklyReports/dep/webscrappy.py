import scrapy
import re

# --- 1. Define Your Data Structures (Items) ---
# Items are like Python dicts but provide structure for your scraped data.

class TeamItem(scrapy.Item):
    """
    Represents a national team.
    """
    team_name = scrapy.Field()
    fifa_rank = scrapy.Field()
    region = scrapy.Field()
    team_page_url = scrapy.Field()
    world_cup_titles = scrapy.Field()
    
class PlayerItem(scrapy.Item):
    """
    Represents a single player.
    """
    player_name = scrapy.Field()
    position = scrapy.Field()
    age = scrapy.Field()
    caps = scrapy.Field()
    goals = scrapy.Field()
    club = scrapy.Field()
    team_name = scrapy.Field() # To link back to the team

# --- 2. Build Your Spider ---
# The spider is the core of Scrapy. It crawls websites and extracts data.

class FifaSpider(scrapy.Spider):
    """
    A spider to scrape FIFA World Cup team and player data.
    
    HOW TO RUN:
    1. Make sure you have Scrapy installed: `pip install scrapy`
    2. Save this file as `fifa_scraper_template.py`
    3. Run it from your terminal: `scrapy runspider fifa_scraper_template.py -o results.csv`
       (This will save the output to a structured CSV file named `results.csv`)
    """
    
    # This name MUST be unique within your Scrapy project
    name = 'fifa_spider'
    
    # --- IMPORTANT ---
    # You MUST change this URL to your target website.
    # A good place to start might be a Wikipedia page, like:
    # 'https://en.wikipedia.org/wiki/2022_FIFA_World_Cup_squads'
    #
    # ALWAYS check the website's 'robots.txt' file first!
    # (e.g., https://en.wikipedia.org/robots.txt)
    # --- /IMPORTANT ---
    start_urls = [
        'https://en.wikipedia.org/wiki/2022_FIFA_World_Cup_squads' # <-- *** REPLACE THIS URL ***
    ]
    
    # Scrapy's main 'engine'. It handles the response from the start_urls.
    def parse(self, response):
        """
        This is the main parsing method.
        It should find all the teams on the starting page and then follow
        the links to each team's individual page.
        """
        
        # --- *** EXAMPLE SELECTOR (NEEDS REPLACEMENT) *** ---
        # This is a HYPOTHETICAL selector. You must inspect your target
        # website using your browser's Developer Tools to find the real one.
        # This example assumes teams are in a 'div' with class 'team-card'.
        #
        # Use `response.css()` or `response.xpath()`
        # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        team_cards = response.css('div.team-card') # <-- *** REPLACE THIS SELECTOR ***
        
        if not team_cards:
            self.log(f"WARNING: No 'div.team-card' found on {response.url}. Did you use the correct URL and selector?")
            self.log("Please inspect the website with browser developer tools to find the correct CSS/XPath selectors.")

        # Loop through each team card found
        for card in team_cards:
            team_item = TeamItem()
            
            # --- *** EXAMPLE SELECTORS (NEEDS REPLACEMENT) *** ---
            # Extract data using the selectors you found.
            # `::text` gets the text. `.get()` gets the first result.
            team_item['team_name'] = card.css('h2.team-name::text').get() # <-- *** REPLACE ***
            team_item['region'] = card.css('span.team-region::text').get() # <-- *** REPLACE ***
            
            # Example of extracting a URL to follow
            team_page_url = card.css('a.team-link::attr(href)').get() # <-- *** REPLACE ***
            
            # Clean up the URL (e.g., if it's a relative path like '/team/brazil')
            if team_page_url:
                team_item['team_page_url'] = response.urljoin(team_page_url)

            # --- --- --- --- --- --- --- --- --- --- --- --- --- ---

            # First, we yield the team item we just scraped.
            # The 'meta' dictionary passes data (like the team_name) to the
            # next callback function (`parse_team_page`).
            yield scrapy.Request(
                url=team_item['team_page_url'], 
                callback=self.parse_team_page,
                meta={'team_name': team_item['team_name']}
            )
            
            # You could also yield the team item here if you want it in your output
            # yield team_item

    def parse_team_page(self, response):
        """
        This method is called for each individual team page.
        It scrapes the player data from that page.
        """
        
        # Get the team_name that we passed from the 'parse' method
        current_team_name = response.meta['team_name']
        
        # --- *** EXAMPLE SELECTOR (NEEDS REPLACEMENT) *** ---
        # This is a HYPOTHETICAL selector for a table of players.
        # You must inspect your target website to find the real one.
        # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        player_rows = response.css('table.player-list tr') # <-- *** REPLACE THIS SELECTOR ***
        
        if not player_rows:
            self.log(f"WARNING: No 'table.player-list tr' found on {response.url}.")
            self.log("Please inspect the website to find the correct player table selector.")

        # Loop through each player row in the table
        for row in player_rows:
            player_item = PlayerItem()
            
            # --- *** EXAMPLE SELECTORS (NEEDS REPLACEMENT) *** ---
            # `td:nth-child(1)` means "the first table cell in the row"
            player_item['player_name'] = row.css('td:nth-child(1)::text').get() # <-- *** REPLACE ***
            player_item['position'] = row.css('td:nth-child(2)::text').get() # <-- *** REPLACE ***
            player_item['age'] = row.css('td:nth-child(3)::text').get() # <-- *** REPLACE ***
            player_item['club'] = row.css('td:nth-child(4) a::text').get() # <-- *** REPLACE ***
            
            # --- --- --- --- --- --- --- --- --- --- --- --- --- ---
            
            # Add the team name for context
            player_item['team_name'] = current_team_name
            
            # Clean up data (example)
            if player_item['age']:
                # Use a try-except block for safer type conversion
                try:
                    player_item['age'] = int(player_item['age'])
                except (ValueError, TypeError):
                    self.log(f"Warning: Could not convert age to int: {player_item['age']}")
                    player_item['age'] = None # Set to None if conversion fails

            # 'yield' the item to send it to your output (e.g., the CSV file)
            yield player_item

# --- 3. (Optional) Configure Settings ---
# To run this script, Scrapy uses default settings.
# For a real project, you would create a `settings.py` file.
# Key settings to add there would be:
#
# ROBOTSTXT_OBEY = True
# DOWNLOAD_DELAY = 1  # 1 second delay between requests (BE POLITE!)
# USER_AGENT = 'WorldCupPredictorBot (your-email@example.com)'
#
# --- --- --- --- --- --- --- --- --- --- ---

