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


class FifaSpider(scrapy.Spider):


    name = 'fifa_spider'
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

        team_cards = response.css('div.team-card') 
        
        if not team_cards:
            self.log(f"WARNING: No 'div.team-card' found on {response.url}. Did you use the correct URL and selector?")
            self.log("Please inspect the website with browser developer tools to find the correct CSS/XPath selectors.")

        # Loop through each team card found
        for card in team_cards:
            team_item = TeamItem()

            team_item['team_name'] = card.css('h2.team-name::text').get()
            team_item['region'] = card.css('span.team-region::text').get()

            # Example of extracting a URL to follow
            team_page_url = card.css('a.team-link::attr(href)').get()

            # Clean up the URL
            if team_page_url:
                team_item['team_page_url'] = response.urljoin(team_page_url)

            yield scrapy.Request(
                url=team_item['team_page_url'], 
                callback=self.parse_team_page,
                meta={'team_name': team_item['team_name']}
            )
            

    def parse_team_page(self, response):
        """
        This method is called for each individual team page.
        It scrapes the player data from that page.
        """
        
        current_team_name = response.meta['team_name']
        
        player_rows = response.css('table.player-list tr')
        
        if not player_rows:
            self.log(f"WARNING: No 'table.player-list tr' found on {response.url}.")
            self.log("Please inspect the website to find the correct player table selector.")

        # Loop through each player row in the table
        for row in player_rows:
            player_item = PlayerItem()
            
            player_item['player_name'] = row.css('td:nth-child(1)::text').get()
            player_item['position'] = row.css('td:nth-child(2)::text').get()
            player_item['age'] = row.css('td:nth-child(3)::text').get()
            player_item['club'] = row.css('td:nth-child(4) a::text').get()
            
            # Add the team name for context
            player_item['team_name'] = current_team_name
            
            if player_item['age']:
                try:
                    player_item['age'] = int(player_item['age'])
                except (ValueError, TypeError):
                    self.log(f"Warning: Could not convert age to int: {player_item['age']}")
                    player_item['age'] = None

            yield player_item