import os
from crewai import Crew
from textwrap import dedent
from agents import ScoutHuntingAgents
from tasks import ScoutHuntingTasks
# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search
from langchain.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

#os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
#os.environ["OPENAI_ORGANIZATION"] = config("OPENAI_ORGANIZATION_ID")

# This is the main class that you will use to define your custom crew.
# You can define as many agents and tasks as you want in agents.py and tasks.py

from dotenv import load_dotenv
load_dotenv()

class TripCrew:
    def __init__(self, origin, cities, date_range, interests):
        self.origin = origin
        self.cities = cities
        self.date_range = date_range
        self.interests = interests

    def run(self):
        # Define your custom agents and tasks in agents.py and tasks.py
        agents = ScoutHuntingAgents()
        tasks = ScoutHuntingTasks()

        # Define your custom agents and tasks here
        expert_film_location_scout_travel_agent = agents.expert_film_location_scout_travel_agent()
        city_scout_expert = agents.city_scout_expert()
        local_movie_location_site_guide = agents.local_movie_location_site_guide()

        # Custom tasks include agent name and variables as input
        plan_itinerary = tasks.plan_itinerary(
           expert_film_location_scout_travel_agent,
           self.cities,
           self.date_range,
           self.interests
        )

        identify_city = tasks.identify_city(
            city_scout_expert,
            self.origin,
            self.cities,
            self.interests,
            self.date_range
        )

        gather_city_info = tasks.gather_city_info(
            local_movie_location_site_guide,
            self.cities,
            self.date_range,
            self.interests
        )

        # Define your custom crew here
        crew = Crew(
            agents=[expert_film_location_scout_travel_agent, 
                    city_scout_expert,
                    local_movie_location_site_guide],
            tasks=[plan_itinerary , 
                   identify_city,
                   gather_city_info],
            verbose=True,
        )

        result = crew.kickoff()
        return result


# This is the main function that you will use to run your custom crew.
if __name__ == "__main__":
    print("## Welcome to Crew AI Template")
    print("-------------------------------")
    origin = input(dedent("""
                            From where will you be travelling from? 
                            """))
    cities = input(dedent("""
                            what are the cities option as destination? 
                            """))
    date_range = input(dedent("""
                            what are the date range for travelling? 
                            """))
    interests = input(dedent("""
                            What are your hobbies and
                            """))
    

    trip_crew = TripCrew(origin,cities,date_range,interests)
    result = trip_crew.run()
    print("\n\n########################")
    print("## Here is you custom crew run result:")
    print("########################\n")
    print(result)
