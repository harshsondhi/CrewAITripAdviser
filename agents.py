from crewai import Agent
from textwrap import dedent
from langchain.llms import OpenAI, Ollama
from langchain_openai import ChatOpenAI

from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools




"""
Creating Agents Cheat Sheet:
- Think like a boss. Work backwards from the goal and think which employee 
    you need to hire to get the job done.
- Define the Captain of the crew who orient the other agents towards the goal. 
- Define which experts the captain needs to communicate with and delegate tasks to.
    Build a top down structure of the crew.

Goal:
- Create a 7-day travel itinerary with detailed per-day plans,
    including budget, packing suggestions, and safety tips.

Captain/Manager/Boss:
- Expert Travel Agent

Employees/Experts to hire:
- City Selection Expert 
- Local Tour Guide


Notes:
- Agents should be results driven and have a clear goal in mind
- Role is their job title
- Goals should actionable
- Backstory should be their resume
"""


# This is an example of how to define custom agents.
# You can define as many agents as you want.
# You can also define custom tasks in tasks.py
class ScoutHuntingAgents:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        self.Ollama = Ollama(model="openhermes")

    def expert_film_location_scout_travel_agent(self):
        return Agent(
            role="Expert in Film location Scout and travel agent",
            backstory=dedent(f"""i am Expert in Film location Scout and travel planning.
                             I have decades of experience making travel itenararies for wide range of locations.
                             Any kind of natural beauty you can imagine.I have access to locations with exotic and unparalleled natural beauties.
                             Like public places like zoo, museum ,mall, buildings to beaches. 
                             """),
            goal=dedent(f"""
                         Create a 7-day travel itinerary with detailed including budget, hotels, packing 
                             suggestion and safety tip
                        """),
            tools=[
                SearchTools.search_internet,
                CalculatorTools.calculate
                ],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def city_scout_expert(self):
        return Agent(
            role="City scout expert",
            backstory=dedent(f"""Expert on analyzing local data and travel destination to pick up ideal destination for movies urban or exotic"""),
            goal=dedent(f"""
                        Select the best city based on weather, natural beauty, great architecture, season, prices and movie location interest
                        """),
             tools=[
                SearchTools.search_internet
                ],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )
        
    def local_movie_location_site_guide(self):
        return Agent(
            role="local movie location site expert",
            backstory=dedent(f"""
                             Since 1980 CAST LOCATIONS has been providing great film locations at affordable rates to a variety of industry 
                             professionals. Working with all the major film studios, television studios, commercial and video companies and 
                             still photographers, CAST has developed a strong working relationship with a broad base of clients.
                             We are a family run business with a dedication to fair, honest and professional work ethics.Knowledgeable local
                             guide with extensive information about the city, it's attraction and customs
                             """),
            goal=dedent(f"""
                        Provide the BEST insights about the seected city
                        """),
            tools=[
                SearchTools.search_internet
                ],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )
